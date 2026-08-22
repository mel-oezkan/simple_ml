
import torch
from torch.utils.data import DataLoader, Subset

from torchvision import datasets
from torchvision.transforms import v2




def prepare_dataloaders(cfg, train_ds, test_ds):
    """Helper function to prepare the dataloaders for training and testing."""
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
    )

    return train_loader, test_loader


def load_datasets(cfg):
    """Helper function to load the FashionMNIST dataset."""

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    if cfg.debug.active and cfg.debug.debug_n:
        training_data = Subset(
            training_data, range(max(cfg.debug.debug_n, cfg.batch_size))
        )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    return training_data, test_data