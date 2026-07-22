import torch.nn as nn
import torch
import numpy as np
from einops import rearrange


class PositionalEmbeddings(nn.Module):
    def __init__(self, model_dim, base: int = 10_000):
        super().__init__()

        self.model_dim = model_dim
        self.dim_subset = torch.arange(model_dim // 2, dtype=torch.float32).reshape(1,-1)
        self.base = float(base)

        self.register_buffer("dim_subset", self.dim_subset)

    def forward(self, positions: torch.Tensor) -> torch.Tensor: 
        # check the shape of positions
        if positions.dim() == 1:
            positions = positions.unsqueeze(-1)
        assert positions.shape[-1] == 1

        pos_sin = torch.sin(positions/(self.base**(2*self.dim_subset/self.model_dim)))
        pos_cos = torch.cos(positions/(self.base**(2*self.dim_subset/self.model_dim)))

        embedding = torch.empty((len(positions), self.model_dim))
        embedding[:, 0::2] = pos_sin
        embedding[:, 1::2] = pos_cos
        return embedding

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, image_size, emb_dim):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        assert image_size % patch_size == 0, f"Non matching patch size given im ({self.image_size}) and patch ({self.patch_size})"

        patch_per_side = self.image_size // self.patch_size
        self.n_patches = patch_per_side ** 2

        self.emb_dim = emb_dim
        self.projector = nn.Conv2d(3, self.emb_dim, self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, image):
        emb = self.projector(image)
        emb = rearrange(emb, "b e h w -> b (h w) e")
        return emb


class ViT(nn.Module):
    def __init__(self):
        super().__init__()

        pass