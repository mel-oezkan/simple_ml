import torch.nn as nn
import torch
from einops import rearrange
import math

from vit.general import MultiHeadAttention, PatchEmbedding, PositionEmbedding


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

        self.pos_emb = PositionEmbedding(emb_dim)
        self.patch_emb = PatchEmbedding(patch_size, image_size, emb_dim)

        self.cls_token = nn.Parameter(torch.zeros(1,1,emb_dim))

        self.vit_blocks = nn.ModuleList([ViT_Block(emb_dim, heads) for _ in range(blocks)])
        self.cls_head = nn.Linear(emb_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image):
        x = self.patch_emb(image)
        _batch_size = x.shape[0]

        # add the cls token to the sequence
        cls = self.cls_token.expand(_batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = x + self.pos_emb(x)

        for block in self.vit_blocks:
            x = block(x)

        return self.cls_head(x[:, 0]) # use cls token to predict class
