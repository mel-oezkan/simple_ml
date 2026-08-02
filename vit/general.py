
import torch.nn as nn
import torch
from einops import rearrange
import math


class SinusoidalEmbedding(nn.Module):
    def __init__(self, model_dim, base: int = 10_000):
        super().__init__()
        assert model_dim % 2 == 0, "model_dim needs to be even"

        self.model_dim = model_dim
        self.base = float(base)

        # 1 / base^(2i/d), precomputed once
        dim_subset = torch.arange(0, model_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (dim_subset / self.model_dim))
        self.register_buffer("inv_freq", inv_freq)


    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (..., 1) -> angles: (..., model_dim // 2)
        angles = positions * self.inv_freq

        # concat sin/cos -> (..., model_dim)
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)


class PositionEmbedding(SinusoidalEmbedding):
    def __init__(self, model_dim, base: int = 10_000):
        super().__init__(model_dim, base)

    def forward(self, x):
        # positions: (...,) -> angles: (..., model_dim // 2)
        seq_len = x.shape[-2]

        positions = torch.arange(
            0, seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(-1)

        return super().forward(positions)

class TimestepEmbedding(SinusoidalEmbedding):
    def __init__(self, model_dim, frequency_dim, base = 10000):
        super().__init__(model_dim, base)

        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim)
        )

    def forward(self, timestep):
        # timestep: (B)
        timestep = timestep.unsqueeze(-1).float()
        freq_emb = super().forward(timestep)  # (B, model_dim)

        return self.mlp(freq_emb)  # (B, model_dim)


class PositionEmbedding2D(SinusoidalEmbedding):
    def __init__(self, model_dim, base: int = 10_000):
        assert model_dim % 4 == 0, "model_dim needs to be divisible by 4 for 2d embeddings"
        # each axis contributes half the channels, so the concat of both is model_dim
        super().__init__(model_dim // 2, base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the 2d positional embeddings for the input tensor x.
        Args:
            x (torch.Tensor): Output of the patch embedding layer, shape (B, N, E)
        Returns:
            pos_emb (torch.Tensor): Positional embeddings, shape (B, N, E)
        """
        # the patches form a square grid, so each axis has sqrt(N) positions
        n_ax_points = int(x.shape[1] ** 0.5)
        assert n_ax_points**2 == x.shape[1], (
            f"expected a square patch grid, got {x.shape[1]} patches"
        )
        axis_positions = torch.arange(
            0, n_ax_points, dtype=torch.float32, device=x.device
        ).unsqueeze(-1) 

        axis_emb = super().forward(axis_positions)  # (num_axis_points, model_dim // 2)

        # adds a dimension and duplicates the embeddings for each axis (to match the size of the other axis)
        rows = axis_emb.unsqueeze(1).expand(n_ax_points, n_ax_points, -1)  # (n_ax_points, n_ax_points, model_dim // 2)
        cols = axis_emb.unsqueeze(0).expand(n_ax_points, n_ax_points, -1)  # (n_ax_points, n_ax_points, model_dim // 2)

        return torch.cat((rows, cols), dim=-1).flatten(0, 1) # (N, model_dim)



POS_EMBEDDINGS = {
    "sinusoidal_1d": PositionEmbedding,
    "sinusoidal_2d": PositionEmbedding2D,
}


def build_pos_embedding(name: str, model_dim: int, n_patches: int) -> nn.Module:
    """Builds the positional embedding selected by `name` (see POS_EMBEDDINGS)."""
    if name not in POS_EMBEDDINGS:
        raise ValueError(
            f"Unknown pos_emb '{name}', expected one of {list(POS_EMBEDDINGS)}"
        )
    return POS_EMBEDDINGS[name](model_dim)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, image_size, emb_dim, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        assert image_size % patch_size == 0, (
            f"Non matching patch size given im ({self.image_size}) and patch ({self.patch_size})"
        )

        patch_per_side = self.image_size // self.patch_size
        self.n_patches = patch_per_side**2

        self.emb_dim = emb_dim
        self.projector = nn.Conv2d(
            in_channels,
            self.emb_dim,
            self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, image):
        emb = self.projector(image)
        emb = rearrange(emb, "b e h w -> b (h w) e")
        return emb

    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, qkv_bias=True):
        super().__init__()

        # check the num heads matches the emb_dim
        assert emb_dim % num_heads == 0

        self.heads = num_heads
        self.emb_dim = emb_dim

        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=qkv_bias)
        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        # input shape: (B, N, E)

        qH, kH, vH = [
            rearrange(i, "B N (h Eh) -> B h N Eh", h=self.heads)
            for i in self.qkv(x).chunk(3, dim=-1)
        ]

        # calculate the attention
        score = (
            qH @ kH.transpose(-2, -1) / math.sqrt(self.emb_dim // self.heads)
        )  # (B h N N)
        out = torch.softmax(score, dim=-1) @ vH  # (B h N N) x (B h N Eh) -> (B h N Eh)

        # combine the heads into a complete tensor
        out = rearrange(out, "B h N Eh -> B N (h Eh)")
        return self.out(out)