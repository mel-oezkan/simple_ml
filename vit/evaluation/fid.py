from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import Inception_V3_Weights, inception_v3

from tqdm import tqdm


def load_backbone_processor(model_name: str = "inception"):
    processor_by_model = {
        "inception": Inception_V3_Weights.DEFAULT.transforms(),
    }

    return processor_by_model[model_name]


""" TODO:
our diffusion model can only generate image in the shape of 28x28 
thus ne need to upscale the image to the inception shape of (3, 299, 299)
"""

f64 = torch.float64


class FID:
    def __init__(self, feature_size: int = 2048, model_backbone: str = "inception"):
        self.model_backbone = model_backbone
        self.feature_size = feature_size

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # for simplicity we simply remove the fc layers. However for g-FID we would
        # import the inception architecture and modify the forward step
        if model_backbone == "inception":
            self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
            self.model.fc = nn.Identity()
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            raise NotImplementedError(":((")

        self.fake_samples = 0
        self.real_samples = 0

        self.mu_fake = torch.zeros(feature_size, dtype=f64)
        self.mu_real = torch.zeros(feature_size, dtype=f64)

        self.centered_prod_fake = np.zeros(
            (feature_size, feature_size), dtype=np.float64
        )
        self.centered_prod_real = np.zeros(
            (feature_size, feature_size), dtype=np.float64
        )

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

        old_prod = torch.as_tensor(old_prod, dtype=f64)
        delta = new_mu - old_mu

        prod_merged = (
            old_prod + new_prod + (old_n * new_n / total) * torch.outer(delta, delta)
        )
        mu_merged = old_mu + delta * (new_n / total)

        self._update_mode(mu_merged, prod_merged, total, mode)

    def compute_features(self, data_loader):
        features = []
        with torch.no_grad():
            for x, y in data_loader:
                features.append(self.model(x).detach().cpu())

        return torch.cat(features, dim=0)

    def feature_statistics(self, features: torch.Tensor, mode: str = "real"):
        # features: (N, 2048)
        features = features.detach().to(device="cpu", dtype=f64)

        mu = torch.mean(features, dim=0)
        centered = features - mu

        product = centered.T @ centered
        self._update(mu, product, features.shape[0], mode=mode)

    def frechet_distance(
        self,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Return a scalar Fréchet distance."""
        sigma_real_t = self.centered_prod_real / (self.real_samples - 1)
        sigma_fake_t = self.centered_prod_fake / (self.fake_samples - 1)

        mu_real = self.mu_real.detach().cpu().numpy()
        mu_fake = self.mu_fake.detach().cpu().numpy()
        sigma_real = sigma_real_t.detach().cpu().numpy()
        sigma_fake = sigma_fake_t.detach().cpu().numpy()

        dist = np.sum((mu_real - mu_fake) ** 2)

        # algorithm is taken from: https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py
        covmean = linalg.sqrtm(sigma_real.dot(sigma_fake))

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
        fid = dist + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * tr_covmean

        return float(np.real(fid))

    def compute_kid(self, feat_real, feat_fake):
        # sanity checks
        if feat_fake.dim() != 2 or feat_real.dim() != 2:
            raise ValueError("features have to be 2dimensional (n_samples, emb_dim)")

        assert feat_real.shape[1] == feat_fake.shape[1], (
            "emb dim between real and fake needs to match"
        )

        def poly_kernel(x, y):
            return torch.pow(torch.inner(x, y) / self.feature_size + 1.0, 3)

        m, n = feat_real.shape[0], feat_fake.shape[0]
        assert m > 1 and n > 1, "Need at least 2 samples for each feature"

        kernel_real = poly_kernel(feat_real, feat_real)
        kernel_fake = poly_kernel(feat_fake, feat_fake)
        kernel_mixed = poly_kernel(feat_real, feat_fake)

        real_term = (kernel_real.sum() - kernel_real.diagonal().sum()) / (m * (m - 1))
        fake_term = (kernel_fake.sum() - kernel_fake.diagonal().sum()) / (n * (n - 1))
        mixed_term = kernel_mixed.mean()

        return real_term + fake_term - 2.0 * mixed_term

    def frechet_distance_from_folder(
        self, folder_real: Path, folder_fake: Path, batch_size: int = 128
    ) -> torch.Tensor:
        # load the images using PIL
        transform_fn = load_backbone_processor(self.model_backbone)
        ds_real = ImageFolder(folder_real, transform=transform_fn)
        ds_fake = ImageFolder(folder_fake, transform=transform_fn)

        data_loader_real = DataLoader(
            ds_real,
            batch_size,
        )
        data_loader_fake = DataLoader(
            ds_fake,
            batch_size,
        )

        print("Computing real features")
        with torch.no_grad():
            for x, _ in tqdm(data_loader_real):
                x = x.to(self.device)
                feat = self.model(x)
                self.feature_statistics(feat, "real")

        print("Computing fake features")
        with torch.no_grad():
            for x, _ in tqdm(data_loader_fake):
                x = x.to(self.device)
                feat = self.model(x)
                self.feature_statistics(feat, "fake")

        print("Returning FID: ")
        return self.frechet_distance()

    def kid_distance_from_folder(
        self, folder_real: Path, folder_fake: Path, batch_size: int = 128
    ) -> torch.Tensor:
        #! this will definetly cause some ood issues since we need to keep track
        #! of the features. We should move all the results to cpu after finishing
        #! the calculation.

        # load the images using PIL
        transform_fn = load_backbone_processor(self.model_backbone)
        ds_real = ImageFolder(folder_real, transform=transform_fn)
        ds_fake = ImageFolder(folder_fake, transform=transform_fn)

        loader_real = DataLoader(ds_real, batch_size)
        loader_fake = DataLoader(ds_fake, batch_size)

        print("Computing real features")
        real_fetures = self.compute_features(loader_real)

        print("Computing fake features")
        fake_fetures = self.compute_features(loader_fake)
        
        print("Returning FID: ")
        return self.compute_kid(real_fetures, fake_fetures)
