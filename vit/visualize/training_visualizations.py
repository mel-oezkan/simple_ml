import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss(save_path: Path, data: dict):
    """Helper function to plot the loss curve."""
    #mean_test_loss

    train_loss = [row.get("mean_loss", None) for row in data]
    test_loss = [row.get("mean_test_loss", None) for row in data]

    plt.figure(figsize=(8, 5))
    plt.title("Training Loss", fontweight="bold", fontsize=14)
    plt.plot(train_loss, label="Training Loss")
    plt.grid("on")

    if any(test_loss):
        plt.plot(test_loss, label="Test Loss", linestyle="--")

    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)

    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()