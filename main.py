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
    transform=v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ]),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True)
    ]),
)


BATCH_SIZE = 128

train_loader = DataLoader(
    training_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


model = Diffusion(
    n_blocks=3,
    emb_dim=64,
    patch_size=4,
    image_size=28,
    out_channels=1,
    mlp_scalar=4,
    T=1000,
)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in tqdm(range(10)):
    for x, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        optimizer.zero_grad()

        noise = torch.randn_like(x)
        pred_noise = model(x, noise)

        loss = nn.functional.mse_loss(pred_noise, noise)

        loss.backward()
        optimizer.step()



    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # show the reverse diffusion process for a sample image
    denoised_sample = model.sample(n=3, device=torch.device("cpu"))

    plt.figure(figsize=(4*3, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(denoised_sample[i].permute(1, 2, 0).detach().numpy())
        plt.axis("off")
    plt.savefig(f"denoised_samples-{epoch + 1}.png")


