import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from scipy import linalg
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

    def feature_statistics(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (N, 2048)
        features = features.to(torch.float32)

        mu = torch.mean(features, dim=0)
        centered = features - mu
        sig = centered.T @ centered / (features.shape[0] - 1)

        return mu, sig

    def frechet_distance(
        self,
        mu_real: torch.Tensor,
        sigma_real: torch.Tensor,
        mu_fake: torch.Tensor,
        sigma_fake: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Return a scalar Fréchet distance."""

        dist = torch.sum((mu_real - mu_fake) ** 2)

        # algorithm is taken from: https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

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
            offset = np.eye(sigma_real.shape[0]) * eps
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return (
            dist.dot(dist)
            + np.trace(sigma_real)
            + np.trace(sigma_fake)
            - 2 * tr_covmean
        )

    def frechet_distance_from_folder(self, folder_path: Path) -> torch.Tensor:
        # load the images using PIL
        processor = load_backbone_processor(self.model_backbone)

        # create a dataloader

        # compute the features

        pass
