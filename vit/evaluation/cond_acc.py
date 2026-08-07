from omegaconf import DictConfig
import torch

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from vit.evaluation.fid import load_backbone_processor
from torchvision import transforms


def acc(pred, label):
    pass

def conditioned_acc():
    pass


def classifier_evaluation(cfg: DictConfig, classifier_path: Path, generated_image_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.load(classifier_path)
    model = model.to(device)
    model.eval()

    ds_generated = ImageFolder(
        generated_image_path,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )

    data_loader_generated = DataLoader(
        ds_generated,
        cfg.batch_size,
    )

    confusion_mat  = torch.zeros((cfg.data.n_classes, cfg.data.n_classes)).to(device)
    with torch.no_grad():
        for (x,y) in data_loader_generated:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred_classes = torch.argmax(pred, dim=1)

            for i,j in zip(y, pred_classes):
                confusion_mat[i][j] += 1

    # for each class compute the class conditional and the average
    total_correct = 0 
    per_class_acc = {}
    for class_index in range(cfg.data.n_classes):
        correct = confusion_mat[class_index][class_index].item()
        total_correct += correct

        per_class_total = confusion_mat[class_index].sum().item()
        per_class_acc[class_index] = correct / per_class_total if per_class_total > 0 else "N/A"

    return total_correct / len(ds_generated), per_class_acc


if __name__ == "__main__":
    model_path = "runs/classifier_experiments/models/cnn/C3/best.pt"
    pass
