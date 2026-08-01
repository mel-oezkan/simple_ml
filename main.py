import json
import uuid
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import datasets
from torchvision.transforms import v2

import matplotlib.pyplot as plt

from vit.diffusion import Diffusion


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


BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 1e-4
SAMPLE_EVERY = 5

config = {
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "sample_every": SAMPLE_EVERY,
    "model": {
        "n_blocks": 3,
        "emb_dim": 128,
        "patch_size": 1,
        "image_size": 28,
        "n_classes": 10,
        "out_channels": 1,
        "mlp_scalar": 4,
        "T": 1000,
    },
}

run_id = str(uuid.uuid4())
run_dir = Path("runs") / run_id
run_dir.mkdir(parents=True, exist_ok=True)
print(f"Run ID: {run_id} (results in {run_dir})")

train_loader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

device = "cuda"
config["device"] = device

with open(run_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

model = Diffusion(**config["model"]).to(device)
model.diff_model = torch.compile(model.diff_model)


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

losses = []

for epoch in range(EPOCHS):
    epoch_losses = []

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        noise = torch.randn_like(x)
        pred_noise = model(x, noise, y)

        loss = nn.functional.mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    mean_loss = sum(epoch_losses) / len(epoch_losses)
    losses.append(
        {
            "epoch": epoch + 1,
            "mean_loss": mean_loss,
            "last_loss": epoch_losses[-1],
        }
    )

    print(f"Epoch {epoch + 1}, Loss: {mean_loss}")

    # write after every epoch so a crashed run keeps its history
    with open(run_dir / "losses.json", "w") as f:
        json.dump(losses, f, indent=2)

    if epoch % SAMPLE_EVERY == 0:
        # show the reverse diffusion process for a sample image
        labels = torch.tensor([3, 6, 9], dtype=torch.long, device=device)
        denoised_sample = model.sample(n=3, device=torch.device("cuda"), y=labels)

        plt.figure(figsize=(4 * 3, 4))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.imshow(denoised_sample[i].permute(1, 2, 0).cpu().detach().numpy())
            plt.title(f"Label: {(i + 1 * 3)}")
            plt.axis("off")
        plt.savefig(run_dir / f"denoised_samples-{epoch + 1}.png")
        plt.close()
