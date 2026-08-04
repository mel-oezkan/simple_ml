from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights, inception_v3


def load_backbone_processor(model_name: str = "inception"):
    processor_by_model = {
        "inception": Inception_V3_Weights.DEFAULT.transforms(),
    }

    return processor_by_model[model_name]


""" TODO:
our diffusion model can only generate image in the shape of 28x28 
thus ne need to upscale the image to the inception shape of (3, 299, 299)
"""


class FID:
    def __init__(self, feature: int = 2048, model_backbone: str = "inception"):
        self.model_backbone = model_backbone

        # for simplicity we simply remove the fc layers. However for g-FID we would
        # import the inception architecture and modify the forward step
        if model_backbone == "inception":
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.model.eval()
        else:
            raise NotImplementedError(":((")

        self.fake_samples = 0
        self.real_samples = 0

        self.mu_fake = 0
        self.mu_real = 0

        self.sigma_fake = np.zeros((feature, feature))
        self.sigma_real = np.zeros((feature, feature))

        self.centered_prod_fake = np.zeros((feature, feature))
        self.centered_prod_real = np.zeros((feature, feature))

    def _load_old(self, mode):
        if mode == "fake":
            return self.centered_prod_fake, self.mu_fake, self.fake_samples

        if mode == "real":
            return self.centered_prod_real, self.mu_real, self.real_samples

        raise ValueError(f"{mode} not valid either use ['real', 'fake']")

    def _update_mode(self, mu, prod, total, mode):
        assert total >= 2, f"number of used samples cannot be < 2: ({total=})"
        new_sigma = prod / (total - 1)

        if mode == "fake":
            self.centered_prod_fake = prod
            self.mu_fake = mu
            self.sigma_fake = new_sigma
            self.fake_samples = total
            return

        if mode == "real":
            self.centered_prod_real = prod
            self.mu_real = mu
            self.sigma_real = new_sigma
            self.real_samples = total
            return
        
        raise ValueError(f"{mode} not valid either use ['real', 'fake']")

    def _update(
        self, new_mu: torch.Tensor, new_prod: torch.Tensor, new_n: int, mode: str
    ):
        old_prod, old_mu, old_n = self._load_old(mode)
        total = old_n + new_n

        delta = (new_mu - old_mu)[:, None]

        prod_merged = (
            old_prod + new_prod + (old_n * new_n) / (total) * (delta @ delta.T)
        )
        mu_merged = old_mu + (delta.unsqueeze(-1) * new_n / total)

        self._update_mode(mu_merged, prod_merged, total, mode)

    def feature_statistics(self, features: torch.Tensor, mode: str = "real"):
        # features: (N, 2048)
        features = features.to(torch.float64)

        mu = torch.mean(features, dim=0)
        centered = features - mu

        product = centered.T @ centered
        self._update(mu, product, features.shape[0], mode=mode)

    def frechet_distance(
        self,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Return a scalar Fréchet distance."""

        dist = torch.sum((self.mu_real - self.mu_fake) ** 2)

        # algorithm is taken from: https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
        covmean, _ = linalg.sqrtm(self.sigma_real.dot(self.sigma_fake), disp=False)

        if not np.isfinite(covmean).all():
            # common cause rank(COV) <= min(D, N-1)
            # Thus when the number of images is smaller than the feature dim
            # the covar mat. is necessarily singular
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)

            # adds (eps) to every eigenvalue:
            offset = np.eye(self.sigma_real.shape[0]) * eps
            covmean = linalg.sqrtm(
                (self.sigma_real + offset).dot(self.sigma_fake + offset)
            )

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (
            dist.dot(dist)
            + np.trace(self.sigma_real)
            + np.trace(self.sigma_fake)
            - 2 * tr_covmean
        )

    def frechet_distance_from_folder(self, folder_path: Path) -> torch.Tensor:
        # load the images using PIL
        data_path = "data/imagenet-10k/imagenet_subtrain"

        transform_fn = load_backbone_processor(self.model_backbone)
        ds = ImageFolder(data_path, transform=transform_fn)

        data_loader = DataLoader(
            ds,
            512,
            shuffle=True,
        )

        # create a dataloader

        # compute the features

        pass
