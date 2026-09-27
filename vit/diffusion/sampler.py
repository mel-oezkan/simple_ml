import torch
from torch import nn

from vit.diffusion.model import Diffusion


def linear_schedule(steps: int, start: float = 1e-4, end: float = 2e-2):
    beta = torch.linspace(start, end, steps)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    return beta, alpha, alpha_bar


def extract(buffer, timestep):
    return buffer[timestep].view(-1, 1, 1, 1)  # (B, 1, 1, 1)


class DDPMSampler(nn.Module):
    def __init__(self, Tmodel: Diffusion, eta: float, T: int, image_size: int, out_channels: int):
        super().__init__()

    


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

    def reverse(self, x_T, y_cond, n_steps, guidance_scale):
        # create the steps and add a last step for ddpm liek schedule
        steps = torch.linspace(self.T - 1, 0, n_steps).long()
        steps = torch.cat([steps, torch.tensor([-1])])

        x_curr = x_T

        for t, t_prev in zip(steps[:-1], steps[1:]):
            # handle the t = -1 cases
            ab = self.model.extract(self.model.alpha_bar, t)
            ab_prev = (
                self.model.extract(self.model.alpha_bar, t_prev)
                if t_prev >= 0
                else torch.ones_like(ab)
            )

            # generate the noise
            z = torch.randn_like(x_curr)

            # get the noise prediction
            batched_t = torch.full(
                (x_curr.shape[0],), t, dtype=torch.long, device=x_curr.device
            )

            if y_cond is not None and guidance_scale != 1.0:
                y_null = torch.full_like(y_cond, self.model.diff_model.n_classes)
                y_in = torch.cat([y_cond, y_null])
                x_in = torch.cat([x_curr, x_curr])
                t_in = torch.cat([batched_t, batched_t])

                if self.model.diff_model.constant_sigma:
                    eps_cond, eps_uncond = self.model.diff_model(
                        x_in, t_in, y_in
                    ).chunk(2)
                else:
                    eps_out, _sig = self.model.diff_model(x_in, t_in, y_in)
                    eps_cond, eps_uncond = eps_out.chunk(2)

                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            else:
                eps = self.model.diff_model(x_curr, batched_t, y_cond)
                if not self.model.diff_model.constant_sigma:
                    eps, _sig = eps

            # calculate the new
            sigma_t = (
                self.eta
                * ((1 - ab_prev) / (1 - ab)).sqrt()
                * ((1 - ab / ab_prev).sqrt())
            )
            x_0 = (x_curr - (1 - ab).sqrt() * eps) / ab.sqrt()
            x_curr = (
                ab_prev.sqrt() * x_0
                + (1 - ab_prev - sigma_t**2).sqrt() * eps
                + sigma_t * z
            )

        return x_curr

    @torch.no_grad()
    def sample(
        self, n: int, n_steps: int, device: str, y=None, guidance_scale: float = 1.0
    ):
        x_T = torch.randn(
            n,
            self.out_channels,
            self.image_size,
            self.image_size,
            device=device,
        )

        return self.reverse(
            x_T, y_cond=y, n_steps=n_steps, guidance_scale=guidance_scale
        )
