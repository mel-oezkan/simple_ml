import json
from pathlib import Path
from typing import Callable

import hydra
import matplotlib.pyplot as plt
from nanoid import generate
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

from vit.model_utils import prepare_model, save_checkpoint
from vit.utils.random import set_seed
from vit.data.utils import load_datasets, prepare_dataloaders
from vit.data.class_mappings import fashion_mnist_mappings




def run_train(model, ema, dataloader, optimizer, criterion, device, num_bins=20):
    """Helper function to perform a single training step."""
    model.train()
    total_loss = 0

    bin_losses = torch.zeros(num_bins, dtype=torch.float32)
    bin_counts = torch.zeros(num_bins, dtype=torch.float32)

    for x, y in tqdm(dataloader, desc="Training: ", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        noise = torch.randn_like(x)
        pred_noise, timesteps = model(x, noise, y) # dim: (B, C, H, W)

        loss = criterion(pred_noise, noise)
        loss.backward()
        optimizer.step()
        ema.update(model)

        # reduce loss to (B)
        per_sample_loss = ((pred_noise - noise) ** 2).mean(dim=(1, 2, 3))
        bin_idx = (timesteps * num_bins) // model.T # generate the indices for the bins (0 to num_bins-1)

        # update
        bin_losses.index_add_(0, bin_idx, per_sample_loss.detach())
        bin_counts.index_add_(0, bin_idx, torch.ones_like(per_sample_loss))

        total_loss += loss.item()

    # prevent division by zero
    mean_bin_loss = bin_losses / bin_counts.clamp_min(1)
    return total_loss / len(dataloader), mean_bin_loss


def run_test(model, dataloader, criterion, device, num_bins=20):
    """Helper function to perform a single test step."""
    model.eval()
    total_loss = 0

    bin_losses = torch.zeros(num_bins, dtype=torch.float32)
    bin_counts = torch.zeros(num_bins, dtype=torch.float32)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            noise = torch.randn_like(x)
            pred_noise, timesteps = model(x, noise, y)

            loss = criterion(pred_noise, noise)
            total_loss += loss.item()

            per_sample_loss = ((pred_noise - noise) ** 2).mean(dim=(1, 2, 3))
            bin_idx = (timesteps * num_bins) // model.T
            bin_losses.index_add_(0, bin_idx, per_sample_loss.detach())
            bin_counts.index_add_(0, bin_idx, torch.ones_like(per_sample_loss))
            
    mean_bin_loss = bin_losses / bin_counts.clamp_min(1)
    return total_loss / len(dataloader), mean_bin_loss


def generate_samples(model, device, plot_path):
    """Simple helper function to generate fashionMNIST samples"""
    model.eval()

    # show the reverse diffusion process for a sample image
    labels = torch.tensor([3, 6, 9], dtype=torch.long, device=device)
    denoised_sample = model.sample(n=3, device=torch.device(device), y=labels)

    plt.figure(figsize=(4 * 3, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        image = denoised_sample[i, 0].add(1).div(2).clamp(0,1)
        plt.imshow(image.cpu(), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Label: {fashion_mnist_mappings[labels[i].item()]}") # map to the class name
        #todo: hardcoded to fashionMNIST (not a current problem since we only support that dataset)

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
    set_seed(cfg.seed)

    # write the config for the run 
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    train_data, test_data = load_datasets(cfg)
    train_dataloader, test_dataloader = prepare_dataloaders(cfg, train_data, test_data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # let matmuls use TF32 tensor cores; also unblocks inductor's
        # fused flash-attention pattern match
        torch.set_float32_matmul_precision("high")

    model, ema, optimizer = prepare_model(cfg, device)

    losses = []
    criterion = nn.MSELoss()

    for epoch in range(cfg.epochs):
        mean_loss, mean_bin_loss = run_train(
            model, ema, train_dataloader, optimizer, criterion, device, cfg.training.num_bins
        )

        mean_test_loss = None
        if not cfg.debug.active and not cfg.debug.get("skip_test", False):
            mean_test_loss, mean_test_bin_loss = run_test(model, test_dataloader, criterion, device, cfg.training.num_bins)

        losses.append(
            {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "mean_bin_loss": mean_bin_loss.tolist(),
                "mean_test_loss": mean_test_loss,
                "mean_test_bin_loss": mean_test_bin_loss.tolist() if mean_test_bin_loss is not None else None,
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
    run_id = generate()
    run_dir = Path(hydra.utils.get_original_cwd()) / "runs" / run_id
    print(f"Run ID: {run_id} (results in {run_dir})")

    if cfg.debug.active:
        cfg.batch_size = cfg.debug.batch_size
        cfg.epochs = cfg.debug.epochs

    train(cfg, run_dir)


if __name__ == "__main__": 
    main()
