from pathlib import Path

from torchvision import transforms

from vit.data.transforms import TransformFashionMNIST
from vit.evaluation.cond_acc import classifier_evaluation


def eval_classification_accuracy(cfg, runs_id_path: Path, generation_id: str) -> dict:
    """Main eval code that computes the FID and conditioned accuracy of a model.

    :param generatio_id: The ID of the generation run to evaluate.
    """
    # ? no seed setting is needed since no rng is used

    # define the storage path for the eval run
    eval_path = runs_id_path 
    generated_image_path = eval_path / generation_id

    acc, cond_acc, confusion_mat = classifier_evaluation(
        cfg,
        classifier_path=Path(cfg.eval.classifier),
        generated_image_path=generated_image_path,
        transform=transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                TransformFashionMNIST.eval,
            ]
        ),
    )

    return {
        "accuracy": acc,
        "conditioned_accuracy": cond_acc,
        "confusion_matrix": confusion_mat,
    }
