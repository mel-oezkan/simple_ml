import torch
from torch import nn

from vit.diffusion import Diffusion


def linear_schedule(steps: int, start: float = 1e-4, end: float = 2e-2):
    beta = torch.linspace(start, end, steps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar


def extract(buffer, timestep):
    return buffer[timestep].view(-1, 1, 1, 1)  # (B, 1, 1, 1)


class DDPMSampler(nn.Module):
    def __init__(self, T):
        super().__init__()

        beta, alpha, alpha_bar = linear_schedule(steps=T)


class DDIMSampler(nn.Module):
    def __init__(self, model: Diffusion, steps, eta, T: int):
        super().__init__()
        self.model = model
        self.T = T
        self.steps = steps

    def reverse(self, x0, steps):
        steps = torch.linspace(self.T - 1, 0, self.steps).long()
        x_curr = x0

        for t, t_prev in zip(steps[:-1], steps[1:]):
            # final output
            if t == 0:
                z = torch.zeros_like(x_curr)
            else:
                z = torch.randn_like(x_curr)

            ab = self.model.extract(self.model.alpha_bar, t)
            ab_prev = self.model.extract(self.model.alpha_bar, t_prev)

            noise_scale = ((1 - ab_prev) / (1 - ab)).sqrt * ((1 - ab) / ab_prev).sqrt

            batched_t = torch.full(
                (x_curr.shape[0],), t, dtype=torch.long, device=x_curr.device
            )
            eps = self.model.diff_model(x_curr, batched_t)

            x_curr = (
                ab_prev * x_curr
                + (1 - ab_prev - noise_scale**2).sqrt * eps
                + noise_scale * z
            )
        
        return x_curr

    @torch.no_grad()
    def sample(self, n: int):
        """Sample from the diffusion model.

        Args:
            n (int): Number of samples to generate.
            device (torch.device): Device to run the sampling on.
            y (torch.Tensor, optional): Class labels for conditional sampling.
        """

        x_T = torch.randn(
            n,
            self.out_channels,
            self.image_size,
            self.image_size,
            device=self.model.device(),
        )
