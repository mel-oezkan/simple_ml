import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_test_loader(data_dir="data", batch_size=128, num_workers=0):
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    return DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_model(model, data_loader, device, loss_fn=None, num_classes=10):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss_fn(logits, labels).item() * batch_size
            total_examples += batch_size

            indices = labels.cpu() * num_classes + predictions.cpu()
            confusion += torch.bincount(
                indices, minlength=num_classes**2
            ).reshape(num_classes, num_classes)

    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty data loader")

    class_total = confusion.sum(dim=1)
    class_correct = confusion.diag()
    per_class_accuracy = torch.where(
        class_total > 0,
        class_correct.float() / class_total,
        torch.zeros_like(class_total, dtype=torch.float32),
    )

    return {
        "loss": total_loss / total_examples,
        "accuracy": class_correct.sum().item() / total_examples,
        "per_class_accuracy": per_class_accuracy.tolist(),
        "class_correct": class_correct.tolist(),
        "class_total": class_total.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")


def print_results(results, class_names=CLASS_NAMES):
    print(f"Loss: {results['loss']:.4f}")
    print(f"Overall accuracy: {results['accuracy']:.2%}\n")

    for name, accuracy, correct, total in zip(
        class_names,
        results["per_class_accuracy"],
        results["class_correct"],
        results["class_total"],
    ):
        print(f"{name:12s}: {accuracy:.2%} ({correct}/{total})")


def evaluate_checkpoint(
    checkpoint_path,
    data_dir="data",
    batch_size=128,
    num_workers=0,
    device=None,
):
    # Imported here to keep evaluate_model reusable from the training script.
    from train_fashion_mnist import build_model

    device = get_device() if device is None else torch.device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_name = checkpoint["model_name"]
    hyperparameters = checkpoint["hyperparameters"]
    model = build_model(
        model_name,
        dropout=hyperparameters["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = get_test_loader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    results = evaluate_model(model, test_loader, device)
    results.update(
        {
            "model_name": model_name,
            "candidate_name": checkpoint["candidate_name"],
            "checkpoint": str(Path(checkpoint_path)),
            "checkpoint_epoch": checkpoint["epoch"],
            "hyperparameters": hyperparameters,
        }
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a Fashion-MNIST experiment checkpoint."
    )
    # Jupyter and Colab inject their kernel connection file as ``-f <path>``.
    parser.add_argument("-f", dest="kernel_connection_file", help=argparse.SUPPRESS)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    return parser.parse_args()


def main():
    args = parse_args()
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    output_path = (
        args.results_dir
        / results["model_name"]
        / results["candidate_name"]
        / "test_evaluation.json"
    )
    save_results(results, output_path)
    print_results(results)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
