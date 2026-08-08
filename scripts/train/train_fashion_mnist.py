import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from scripts.train.eval_fashion_mnist import evaluate_model, get_device, save_results
from vit.models.classifier import MLPClassifier, CNNClassifier

from vit.data.transforms import TransformFashionMNIST
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=2,
        in_channels=1,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(embed_dim),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        x = self.patch_embedding(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.embedding_dropout(x + self.position_embedding)
        x = self.encoder(x)
        return self.classifier(x[:, 0])


HYPERPARAMETER_CANDIDATES = {
    "mlp": {
        "M1": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "M2": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "M3": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.20,
            "label_smoothing": 0.0,
        },
        "M4": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "dropout": 0.20,
            "label_smoothing": 0.05,
        },
        "M5": {
            "learning_rate": 3e-3,
            "weight_decay": 1e-4,
            "dropout": 0.10,
            "label_smoothing": 0.05,
        },
    },
    "cnn": {
        "C1": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "C2": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "C3": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-3,
            "dropout": 0.20,
            "label_smoothing": 0.0,
        },
        "C4": {
            "learning_rate": 3e-3,
            "weight_decay": 1e-4,
            "dropout": 0.20,
            "label_smoothing": 0.05,
        },
        "C5": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "dropout": 0.30,
            "label_smoothing": 0.05,
        },
    },
    "vit": {
        "V1": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-2,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "V2": {
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "dropout": 0.10,
            "label_smoothing": 0.0,
        },
        "V3": {
            "learning_rate": 3e-4,
            "weight_decay": 1e-3,
            "dropout": 0.05,
            "label_smoothing": 0.0,
        },
        "V4": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-2,
            "dropout": 0.10,
            "label_smoothing": 0.05,
        },
        "V5": {
            "learning_rate": 3e-4,
            "weight_decay": 5e-2,
            "dropout": 0.20,
            "label_smoothing": 0.05,
        },
    },
}


def build_model(name, dropout=0.1):
    if name == "mlp":
        return MLPClassifier(dropout=dropout)
    if name == "cnn":
        return CNNClassifier(dropout=dropout)
    if name == "vit":
        return VisionTransformer(dropout=dropout)
    raise ValueError(f"Unknown model {name!r}; choose 'mlp', 'cnn', or 'vit'")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_validation_data(data_dir, validation_size, split_seed):
    train_transform = TransformFashionMNIST.train
    evaluation_transform = TransformFashionMNIST.eval

    augmented_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    evaluation_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=evaluation_transform,
    )

    if not 0 < validation_size < len(augmented_data):
        raise ValueError("validation_size must be between 1 and the dataset size")

    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(augmented_data), generator=generator).tolist()
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    return (
        Subset(augmented_data, training_indices),
        Subset(evaluation_data, validation_indices),
    )


def get_train_validation_loaders(
    training_data,
    validation_data,
    batch_size,
    num_workers,
    seed,
    device,
):
    common_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        training_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common_options,
    )
    validation_loader = DataLoader(
        validation_data,
        shuffle=False,
        **common_options,
    )
    return train_loader, validation_loader


def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_checkpoint(
    path,
    model,
    model_name,
    candidate_name,
    hyperparameters,
    epoch,
    validation_metrics,
    parameter_count,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "candidate_name": candidate_name,
            "hyperparameters": hyperparameters,
            "epoch": epoch,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "parameter_count": parameter_count,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def run_candidate(
    model_name,
    candidate_name,
    hyperparameters,
    training_data,
    validation_data,
    args,
    device,
):
    result_dir = args.results_dir / model_name / candidate_name
    model_dir = args.models_dir / model_name / candidate_name
    history_path = result_dir / "history.json"
    summary_path = result_dir / "summary.json"
    checkpoint_path = model_dir / "best.pt"

    if summary_path.exists() and not args.overwrite:
        print(f"Skipping completed run {model_name}/{candidate_name}")
        return json.loads(summary_path.read_text())

    set_seed(args.seed)
    train_loader, validation_loader = get_train_validation_loaders(
        training_data=training_data,
        validation_data=validation_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=device,
    )

    model = build_model(
        model_name,
        dropout=hyperparameters["dropout"],
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=hyperparameters["label_smoothing"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )

    history = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0

    progress = tqdm(
        range(1, args.epochs + 1),
        desc=f"{model_name.upper()}/{candidate_name}",
        unit="epoch",
    )
    for epoch in progress:
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_loss_fn,
            device,
        )
        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_metrics)

        improved = validation_metrics["loss"] < (
            best_validation_loss - args.minimum_delta
        )
        if improved:
            best_epoch = epoch
            best_validation_loss = validation_metrics["loss"]
            best_validation_accuracy = validation_metrics["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                model_name,
                candidate_name,
                hyperparameters,
                epoch,
                validation_metrics,
                parameter_count,
            )
        else:
            epochs_without_improvement += 1

        progress.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_loss=f"{validation_metrics['loss']:.4f}",
            val_acc=f"{validation_metrics['accuracy']:.2%}",
        )
        progress.write(
            f"{model_name.upper()}/{candidate_name} epoch {epoch}/{args.epochs} | "
            f"train_loss: {train_metrics['loss']:.4f} | "
            f"train_acc: {train_metrics['accuracy']:.2%} | "
            f"val_loss: {validation_metrics['loss']:.4f} | "
            f"val_acc: {validation_metrics['accuracy']:.2%}"
        )

        save_results(
            {
                "status": "running",
                "model_name": model_name,
                "candidate_name": candidate_name,
                "hyperparameters": hyperparameters,
                "parameter_count": parameter_count,
                "history": history,
            },
            history_path,
        )

        if epochs_without_improvement >= args.patience:
            progress.write(
                f"Early stopping {model_name.upper()}/{candidate_name} "
                f"after epoch {epoch}"
            )
            break

    summary = {
        "model_name": model_name,
        "candidate_name": candidate_name,
        "hyperparameters": hyperparameters,
        "parameter_count": parameter_count,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "max_validation_accuracy": max(
            metrics["validation_accuracy"] for metrics in history
        ),
        "checkpoint": str(checkpoint_path),
    }
    save_results(
        {
            "status": "complete",
            **summary,
            "history": history,
        },
        history_path,
    )
    save_results(summary, summary_path)
    return summary


def run_hyperparameter_evaluation(args):
    device = get_device() if args.device is None else torch.device(args.device)
    print(f"Using device: {device}")

    training_data, validation_data = get_train_validation_data(
        data_dir=args.data_dir,
        validation_size=args.validation_size,
        split_seed=args.split_seed,
    )

    all_summaries = []
    for model_name in args.models:
        model_summaries = []
        for candidate_name, hyperparameters in HYPERPARAMETER_CANDIDATES[
            model_name
        ].items():
            summary = run_candidate(
                model_name=model_name,
                candidate_name=candidate_name,
                hyperparameters=hyperparameters,
                training_data=training_data,
                validation_data=validation_data,
                args=args,
                device=device,
            )
            model_summaries.append(summary)
            all_summaries.append(summary)

        best_candidate = min(
            model_summaries,
            key=lambda summary: summary["best_validation_loss"],
        )
        save_results(
            {
                "model_name": model_name,
                "selection_metric": "best_validation_loss",
                "best_candidate": best_candidate,
                "candidates": model_summaries,
            },
            args.results_dir / model_name / "summary.json",
        )

    best_run = min(
        all_summaries,
        key=lambda summary: summary["best_validation_loss"],
    )
    save_results(
        {
            "selection_metric": "best_validation_loss",
            "best_run": best_run,
            "runs": all_summaries,
        },
        args.results_dir / "summary.json",
    )
    print(
        f"Best run: {best_run['model_name'].upper()}/"
        f"{best_run['candidate_name']} with validation loss "
        f"{best_run['best_validation_loss']:.4f}"
    )
    print(f"Checkpoint: {best_run['checkpoint']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Fashion-MNIST MLP, CNN, and ViT hyperparameter evaluations."
    )
    # Jupyter and Colab inject their kernel connection file as ``-f <path>``.
    parser.add_argument("-f", dest="kernel_connection_file", help=argparse.SUPPRESS)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(HYPERPARAMETER_CANDIDATES),
        default=list(HYPERPARAMETER_CANDIDATES),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    run_hyperparameter_evaluation(parse_args())


if __name__ == "__main__":
    main()
