import torch.nn as nn
import torch
from einops import rearrange


class PositionalEmbeddings(nn.Module):
    def __init__(self, model_dim, base: int = 10_000):
        super().__init__()

        self.model_dim = model_dim
        self.base = float(base)

        # 1 / base^(2i/d), precomputed once
        dim_subset = torch.arange(model_dim // 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (2 * dim_subset / self.model_dim))
        self.register_buffer("inv_freq", inv_freq)


    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (...,) -> angles: (..., model_dim // 2)
        angles = positions.float().unsqueeze(-1) * self.inv_freq

        # interleave sin/cos -> (..., model_dim)
        embedding = torch.stack((
            torch.sin(angles), torch.cos(angles)), dim=-1)
        return embedding.flatten(-2)

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, image_size, emb_dim, in_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        assert image_size % patch_size == 0, f"Non matching patch size given im ({self.image_size}) and patch ({self.patch_size})"

        patch_per_side = self.image_size // self.patch_size
        self.n_patches = patch_per_side ** 2

        self.emb_dim = emb_dim
        self.projector = nn.Conv2d(in_channels, self.emb_dim, self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, image):
        emb = self.projector(image)
        emb = rearrange(emb, "b e h w -> b (h w) e")
        return emb


class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        pass