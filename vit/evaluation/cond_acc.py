from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from vit.models.classifier import load_cnn_classifier


def acc(pred, label):
    if pred.ndim > label.ndim:
        pred = pred.argmax(dim=-1)

    return (pred == label).float().mean()


def conditioned_acc(cfg, labels: torch.Tensor, predictions: torch.Tensor):
    device = predictions.device
    confusion_mat = torch.zeros(
        (cfg.data.n_classes, cfg.data.n_classes),
        dtype=torch.long,
        device=device
    )

    for class_idx in range(cfg.data.n_classes):
        # indx all labels 
        rows_preds = predictions[labels == class_idx]

        confusion_mat[class_idx] = torch.bincount(
            rows_preds.long(),
            minlength=cfg.data.n_classes,
        )
    class_totals = confusion_mat.sum(dim=1)
    per_class_accuracy = confusion_mat.diag() / class_totals

    return confusion_mat, per_class_accuracy


def classifier_evaluation(
    cfg: DictConfig,
    classifier_path: Path,
    generated_dataset: Dataset,
) -> tuple[float, dict[int, float | str], list[list[int]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_cnn_classifier(
        classifier_path,
        num_classes=cfg.data.n_classes,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    data_loader_generated = DataLoader(
        generated_dataset,
        cfg.batch_size,
    )

    confusion_mat = torch.zeros(
        (cfg.data.n_classes, cfg.data.n_classes),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        for x, y in data_loader_generated:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred_classes = torch.argmax(pred, dim=1)

            for i, j in zip(y, pred_classes):
                confusion_mat[i][j] += 1

    # for each class compute the class conditional and the average
    total_correct = 0
    per_class_acc = {}
    for class_index in range(cfg.data.n_classes):
        correct = confusion_mat[class_index][class_index].item()
        total_correct += correct

        per_class_total = confusion_mat[class_index].sum().item()
        per_class_acc[class_index] = (
            correct / per_class_total if per_class_total > 0 else "N/A"
        )

    return (
        total_correct / len(generated_dataset),
        per_class_acc,
        confusion_mat.cpu().tolist(),
    )


if __name__ == "__main__":
    model_path = "runs/classifier_experiments/models/cnn/C3/best.pt"
    pass
