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
    def __init__(
        self, model: Diffusion, eta: float, T: int, image_size: int, out_channels: int
    ):
        super().__init__()
        self.model = model
        self.eta = eta
        self.T = T
        self.image_size = image_size
        self.out_channels = out_channels

    def reverse(self, x0, n_steps):
        # create the steps and add a last step for ddpm liek schedule
        steps = torch.linspace(self.T - 1, 0, n_steps).long()
        steps = torch.cat([steps, torch.tensor([-1])])

        x_curr = x0

        for t, t_prev in zip(steps[:-1], steps[1:]):
            # handle the t = -1 cases
            ab = self.model.extract(self.model.alpha_bar, t)
            ab_prev = (
                self.model.extract(self.model.alpha_bar, t_prev)
                if t_prev >= 0
                else torch.ones_like(ab)
            )
            
            # generate the noise 
            z = torch.rand_like(x_curr)

            # get the noise prediction
            batched_t = torch.full(
                (x_curr.shape[0],), t, dtype=torch.long, device=x_curr.device
            )
            eps = self.model.diff_model(x_curr, batched_t)

            # calculate the new 
            sigma_t = self.eta * ((1 - ab_prev) / (1 - ab)).sqrt() * ((1 - ab / ab_prev).sqrt())
            x_0 = (x_curr - (1 - ab).sqrt() * eps) / ab.sqrt()
            x_curr = (
                ab_prev.sqrt() * x_0
                + (1 - ab_prev - sigma_t**2).sqrt() * eps
                + sigma_t * z
            )

        return x_curr

    @torch.no_grad()
    def sample(self, n: int, n_steps: int, device: str):
        x_T = torch.randn(
            n,
            self.out_channels,
            self.image_size,
            self.image_size,
            device=device,
        )

        return self.reverse(x_T, n_steps)
