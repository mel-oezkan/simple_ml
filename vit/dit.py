import torch
import torch.nn as nn

from einops import rearrange

from vit.general import MultiHeadAttention, PatchEmbedding, TimestepEmbedding, build_pos_embedding


class DiT_Block(nn.Module):
    def __init__(self, emb_dim, attn_heads=4, mlp_scalar=4):
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
            nn.GELU(approximate='tanh'), 
            nn.Linear(scaled_dim, emb_dim)
        )

    def forward(self, x, cond):
        # cond: (B, emb_dim)
        gam1, bet1, alp1, gam2, bet2, alp2 = torch.chunk(
            self.conditions(cond).unsqueeze(1), 6, dim=-1
        )

        path1 = self.norm1(x) * (1 + gam1) + bet1
        path1 = self.mha(path1)
        path1 = path1 * alp1

        x = x + path1

        path2 = self.norm2(x) * (1 + gam2) + bet2
        path2 = self.mlp(path2)
        path2 = path2 * alp2

        x = x + path2

        return x # (B, N, emb_dim)

class DIT_Final(nn.Module):
    def __init__(self, emb_dim, patch_size, image_size, out_channels=3):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.ln = nn.LayerNorm(emb_dim, elementwise_affine=False)

        self.modulation = nn.Linear(emb_dim, emb_dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        self.out_head = nn.Linear(emb_dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, latent, cond):
        # latent: (B, N, emb_dim), cond: (B, emb_dim)
        scale, shift = torch.chunk(self.modulation(cond).unsqueeze(1), 2, dim=-1)

        out = self.ln(latent) * (1 + scale) + shift
        out = self.out_head(out)

        out = rearrange(out, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
            h = self.image_size // self.patch_size,
            w = self.image_size // self.patch_size,
            c=self.out_channels,
            p1=self.patch_size,
            p2=self.patch_size
        )

        return out


class DiT(nn.Module):
    def __init__(
        self, 
        n_blocks, 
        emb_dim, 
        patch_size, 
        image_size, 
        n_classes = None,
        class_dropout = 0.1,
        out_channels=3, 
        mlp_scalar=4,
        constant_sigma=True,
        frequency_dim=256,
        pos_emb="sinusoidal_1d",

    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        self.constant_sigma = constant_sigma
        if not constant_sigma:
            out_channels = out_channels * 2

        self.out_channels = out_channels

        # create the class tokens for conditional diffusion
        self.n_classes = n_classes
        self.class_dropout = class_dropout
        if n_classes is not None:
            # +1 slot is the "null" class used for dropout / CFG
            self.label_emb = nn.Embedding(n_classes + 1, emb_dim)
            nn.init.normal_(self.label_emb.weight, std=0.02)

        self.patch_emb = PatchEmbedding(patch_size, image_size, emb_dim, in_channels=out_channels)
        self.pos_emb = build_pos_embedding(pos_emb, emb_dim, self.patch_emb.n_patches)
        self.time_emb = TimestepEmbedding(emb_dim, frequency_dim=emb_dim)

        self.forward_blocks = nn.ModuleList(
            [DiT_Block(emb_dim, mlp_scalar=mlp_scalar) for _ in range(n_blocks)]
        )        

        self.final = DIT_Final(emb_dim, patch_size, image_size, out_channels)


    def forward(self, latent, t, y=None):

        if self.n_classes is not None:
            if y is None:
                y = torch.full(
                    (latent.shape[0],),
                    self.n_classes,  # null token
                    device=latent.device,
                    dtype=torch.long
                )


            if self.training and self.class_dropout > 0:
                drop_prob = torch.rand(
                    y.shape, 
                    device=y.device
                )

                y = torch.where(
                    drop_prob < self.class_dropout, 
                    self.n_classes, # sets index to null token
                    y    # keeps the original class label
                )

        cond_emb = self.time_emb(t)
        if self.n_classes is not None:
            cond_emb = cond_emb + self.label_emb(y)

        latent_emb = self.patch_emb(latent)
        if self.pos_emb is not None:
            latent_emb = latent_emb + self.pos_emb(latent_emb)

        for block in self.forward_blocks:
            latent_emb = block(latent_emb, cond_emb)

        out = self.final(latent_emb, cond_emb)
        if self.constant_sigma:
            # out: (B, C, H, W)
            return out

        # out: (B, C*2, H, W) learnes the sigma
        return torch.chunk(out, 2, dim=1) # out, sig
