import torch.nn as nn
import torch
from einops import rearrange
import math


class PositionalEmbeddings(nn.Module):
    def __init__(self, model_dim, base: int = 10_000):
        super().__init__()
        assert model_dim % 2 == 0, "model_dim needs to be even"

        self.model_dim = model_dim
        self.base = float(base)

        # 1 / base^(2i/d), precomputed once
        dim_subset = torch.arange(0, model_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (self.base ** (dim_subset / self.model_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # positions: (...,) -> angles: (..., model_dim // 2)
        seq_len = x.shape[-2]

        positions = torch.arange(
            0, seq_len, dtype=torch.float32, device=x.device
        ).unsqueeze(-1)
        angles = positions * self.inv_freq

        # interleave sin/cos -> (..., model_dim)
        embedding = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
        return embedding.flatten(-2)


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


class ViT_Block(nn.Module):
    def __init__(self, emb_dim, num_heads, mlp_scalar=4):
        super().__init__()

        self.norm1 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(num_heads, emb_dim)

        scaled_dim = mlp_scalar * emb_dim
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, scaled_dim), nn.GELU(), nn.Linear(scaled_dim, emb_dim)
        )

    def forward(self, x):
        path1 = self.norm1(x)
        path1 = self.mha(path1)

        x = x + path1

        path2 = self.norm2(x)
        path2 = self.mlp(path2)

        return x + path2


class ViT(nn.Module):
    def __init__(self, emb_dim, heads, blocks, num_classes, patch_size=32, image_size=64):
        super().__init__()

        self.pos_emb = PositionalEmbeddings(emb_dim)
        self.patch_emb = PatchEmbedding(patch_size, image_size, emb_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_dim))

        self.vit_blocks = nn.ModuleList([ViT_Block(emb_dim, heads) for _ in range(blocks)])
        self.cls_head = nn.Linear(emb_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image):
        x = self.pos_emb(image)
        _batch_size = x.shape[0]

        # add the cls token to the sequence
        cls = self.cls_token.expand(_batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_emb(x)

        for block in self.vit_blocks:
            x = block(x)

        return self.cls_head(x[:, 0]) # use cls token to predict class
