
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
    def __init__(self, model_dim):
        super().__init__(model_dim)

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