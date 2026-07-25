import torch.nn as nn
import torch
from einops import rearrange
import math

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
        score = qH @ kH.transpose(-2, -1) / math.sqrt(self.emb_dim // self.heads) # (B h N N)
        out = torch.softmax(score, dim=-1) @ vH  # (B h N N) x (B h N Eh) -> (B h N Eh)

        # combine the heads into a complete tensor
        out = rearrange(out, "B h N Eh -> B N (h Eh)")
        return self.out(out)


class ViT_Block(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_scalar=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(num_heads, emb_dim)

        scaled_dim = mlp_scalar * emb_dim
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, scaled_dim),
            nn.GELU(),
            nn.Linear(scaled_dim, emb_dim)
        )

    def forward(self, x):

        path1 = self.norm1(x)
        path1 = self.mha(path1)

        x = x + path1

        path2 = self.norm2(x)
        path2 = self.mlp(path2)

        return x + path2
    
class ViT(nn.Module):
    def __init__(self, emb_dim, heads, blocks, image_size):
        super().__init__()

        self.pos_emb = PositionalEmbeddings(emb_dim)
        self.
        pass