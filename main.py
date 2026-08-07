import json
from typing import Callable
import uuid
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from vit.diffusion import Diffusion
from vit.ema import EMA
from vit.model_utils import save_checkpoint


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


def prepare_dataloaders(cfg, train_ds, test_ds):
    """Helper function to prepare the dataloaders for training and testing."""

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, test_loader


def run_train(model, ema, dataloader, optimizer, criterion, device):
    """Helper function to perform a single training step."""
    model.train()
    total_loss = 0
    for x, y in tqdm(dataloader, desc="Training: ", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        noise = torch.randn_like(x)
        pred_noise = model(x, noise, y)

        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        ema.update(model)

        total_loss += loss.item()

    return total_loss / len(dataloader)


def run_test(model, dataloader, criterion, device):
    """Helper function to perform a single test step."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            noise = torch.randn_like(x)
            pred_noise = model(x, noise, y)

            loss = criterion(pred_noise, noise)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def generate_samples(model, device, plot_path):
    model.eval()
    # show the reverse diffusion process for a sample image
    labels = torch.tensor([3, 6, 9], dtype=torch.long, device=device)
    denoised_sample = model.sample(n=3, device=torch.device(device), y=labels)

    plt.figure(figsize=(4 * 3, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(denoised_sample[i].permute(1, 2, 0).cpu().detach().numpy())
        plt.title(f"Label: {((i + 1) * 3)}")
        plt.axis("off")

    plt.savefig(plot_path)
    plt.close()


def train(
    cfg: DictConfig, run_dir: Path, on_epoch_end: Callable[[], None] | None = None
):
    """Run the full training loop, writing artifacts into run_dir.

    Kept free of hydra runtime state so it can be called from Modal too.

    args:
        cfg (DictConfig): Hydra condif converted to dict
        run_dir (pathlib.Path): Output directory for the run
        on_epoch_end (callable | None): hook function to run on the end of epoch (defaults to None)
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = load_datasets(cfg)
    train_dataloader, test_dataloader = prepare_dataloaders(cfg, train_data, test_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # let matmuls use TF32 tensor cores; also unblocks inductor's
        # fused flash-attention pattern match
        torch.set_float32_matmul_precision("high")

    model = Diffusion(**cfg["model"]).to(device)
    model.diff_model = torch.compile(model.diff_model)

    ema = EMA(model, cfg.ema.decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    losses = []
    criterion = nn.MSELoss()

    for epoch in range(cfg.epochs):
        mean_loss = run_train(
            model, ema, train_dataloader, optimizer, criterion, device
        )

        mean_test_loss = None
        if not cfg.debug.get("skip_test", False):
            mean_test_loss = run_test(model, test_dataloader, criterion, device)

        losses.append(
            {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "mean_test_loss": mean_test_loss,
            }
        )

        print(f"Epoch {epoch + 1}, Loss: {mean_loss}, Test Loss: {mean_test_loss}")

        # write after every epoch so a crashed run keeps its history
        with open(run_dir / "losses.json", "w") as f:
            json.dump(losses, f, indent=2)

        if epoch % cfg.sample_every == 0:
            plot_path = run_dir / f"denoised_samples-{epoch + 1}.png"
            with ema.averaged(model):
                generate_samples(model, device, plot_path)

        if on_epoch_end:
            # when run with modal this will commti to the volume
            on_epoch_end()

    # final results
    plot_path = run_dir / "denoised_samples-final.png"
    with ema.averaged(model):
        generate_samples(model, device, plot_path)

    # store the final weights in runs
    save_checkpoint(cfg, model, ema, epoch, optimizer, run_dir)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # hydra calls main with the config only, so the run dir is built here
    run_id = str(uuid.uuid4())
    run_dir = Path(hydra.utils.get_original_cwd()) / "runs" / run_id
    print(f"Run ID: {run_id} (results in {run_dir})")

    if cfg.debug.active:
        cfg.batch_size = cfg.debug.batch_size
        cfg.epochs = cfg.debug.epochs

    train(cfg, run_dir)


if __name__ == "__main__":
    main()