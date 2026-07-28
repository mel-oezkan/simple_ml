import torch
import torch.nn as nn

from vit.general import MultiHeadAttention

class DIT(nn.Module):
    def __init__(self, emb_dim, attn_heads=5, mlp_scalar = 4):
        super().__init__()

        self.conditions = nn.Linear(emb_dim, emb_dim * 6)
        nn.init.zeros_(self.conditions.weight)
        nn.init.zeros_(self.conditions.bias)

        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False)

        self.mha = MultiHeadAttention(attn_heads, emb_dim)

        scaled_dim = emb_dim * mlp_scalar
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, scaled_dim),
            nn.SiLU(),
            nn.Linear(scaled_dim, emb_dim)
        )

    def forward(self, x, cond):
        # cond: (B, emb_dim)
        alp1, bet1, gam1, alp2, bet2, gam2 = torch.chunk(
            self.conditions(cond).unsqueeze(1), 6, dim=-1
        )

        path1 = self.norm1(x) * (1 + gam1) + bet1
        path1 = self.mha(path1)
        path1 = path1 + alp1.unsqueeze(1)

        x = x + path1

        path2 = self.norm2(x) * (1 + gam2) + bet2
        path2 = self.mlp(path2)
        path2 = path2 + alp2.unsqueeze(1)

        x = x + path2

        return x
