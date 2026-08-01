import torch
import torch.nn as nn

from vit.dit import DiT


def linear_schedule(steps: int, start: float = 1e-4, end: float = 2e-2):
    beta = torch.linspace(start, end, steps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar


# ! incorrect
# def cosine_schedule(steps: int, s: float = 0.008):
#     """Compute the cosine schedule for diffusion models.
#     Args:
#         steps (int): Number of diffusion steps.
#         s (float): Small offset to prevent singularities.
#     """
#     t = torch.linspace(0, steps, steps + 1) / steps

#     cycle = math.pi / 2
#     period = 1 + s
#     f_t = torch.cos(((t + s) / period) * cycle) ** 2
#     alpha_bar = f_t[1:] / f_t[:-1]

#     beta = 1 - alpha_bar
#     alpha = 1 - beta

#     return beta, alpha, alpha_bar


class Diffusion(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        emb_dim: int,
        patch_size: int,
        image_size: int,
        n_classes: int = None,
        out_channels: int = 3,
        mlp_scalar=4,
        T=1000,
    ):
        super().__init__()

        self.T = T
        self.image_size = image_size
        self.out_channels = out_channels

        beta, alpha, alpha_bar = linear_schedule(steps=T)
        # self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta_sqrt", torch.sqrt(beta))
        self.register_buffer("alpha_sqrt", torch.sqrt(alpha))
        self.register_buffer("alpha_bar_sqrt", torch.sqrt(alpha_bar))
        self.register_buffer("one_minus_alpha_sqrt", torch.sqrt(1 - alpha))
        self.register_buffer("one_minus_alpha_bar_sqrt", torch.sqrt(1 - alpha_bar))

        self.diff_model = DiT(
            n_blocks,
            emb_dim,
            patch_size,
            image_size,
            n_classes=n_classes,
            out_channels=out_channels,
            mlp_scalar=mlp_scalar,
        )

    def extract(self, buf, t):
        return buf[t].view(-1, 1, 1, 1)  # (B, 1, 1, 1)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device, y=None) -> torch.Tensor:
        """Sample from the diffusion model.

        Args:
            n (int): Number of samples to generate.
            device (torch.device): Device to run the sampling on.
            y (torch.Tensor, optional): Class labels for conditional sampling.
        """
        device = device or next(self.parameters()).device

        x_T = torch.randn(
            n, self.out_channels, self.image_size, self.image_size, device=device
        )

        return self.reverse(x_T, y=y)

    def reverse(self, x_curr: torch.Tensor, y=None) -> torch.Tensor:
        assert x_curr.dim() == 4, "x_curr must be a 4D tensor (B, C, H, W)"

        with torch.no_grad():
            for t in torch.arange(self.T - 1, -1, -1, dtype=torch.long):
                if t == 0:
                    z = torch.zeros_like(x_curr)
                else:
                    z = torch.randn_like(x_curr)

                batched_t = torch.full(
                    (x_curr.shape[0],), t, 
                    dtype=torch.long, 
                    device=x_curr.device
                )

                s = 1 / self.extract(self.alpha_sqrt, batched_t)
                frac = (1 - self.extract(self.alpha, batched_t)) / self.extract(
                    self.one_minus_alpha_bar_sqrt, batched_t
                )
                sigma = self.extract(self.beta_sqrt, batched_t)

                x_curr = (
                    s * (x_curr - frac * self.diff_model(x_curr, batched_t, y)) + sigma * z
                )

        return x_curr

    def forward(self, x_0: torch.Tensor, noise: torch.Tensor):
        """Training step for the diffions model.

        Args:
            x_0 (torch.Tensor): clean latent/image
            noise (torch.Tensor): Noise/eps naming was selected for simplicity

        Returns:
            torch.Tensor: predicted noise
        """

        t = torch.randint(0, self.T, (x_0.shape[0],), device=x_0.device).long()
        u = self.extract(self.alpha_bar_sqrt, t)
        v = self.extract(self.one_minus_alpha_bar_sqrt, t)

        x_t = u * x_0 + v * noise

        return self.diff_model(x_t, t)
