from pathlib import Path

from torch.utils.data import Dataset

from vit.evaluation.cond_acc import classifier_evaluation


def eval_classification_accuracy(cfg, generated_dataset: Dataset) -> dict:
    """Compute classifier accuracy for the selected generated samples."""
    acc, cond_acc, confusion_mat = classifier_evaluation(
        cfg,
        classifier_path=Path(cfg.eval.classifier),
        generated_dataset=generated_dataset,
    )

    return {
        "accuracy": acc,
        "conditioned_accuracy": cond_acc,
        "confusion_matrix": confusion_mat,
    }
