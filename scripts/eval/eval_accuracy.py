from pathlib import Path

from vit.evaluation.cond_acc import classifier_evaluation
from vit.data.transforms import TransformFashionMNIST


def eval_classification_accuracy(cfg, runs_path: Path, generation_id: str) -> dict:
    """Main eval code that computes the FID and conditioned accuracy of a model.

    :param generatio_id: The ID of the generation run to evaluate.
    """
    # ? no seed setting is needed since no rng is used

    # determine the run_id from the checkpoint path
    configured_checkpoint = Path(cfg.eval.checkpoint_path)
    run_id = configured_checkpoint.parent.name

    # define the storage path for the eval run
    eval_path = runs_path / run_id
    generated_image_path = eval_path / generation_id

    acc, cond_acc, confusion_mat = classifier_evaluation(
        cfg,
        classifier_path=Path(cfg.eval.classifier),
        generated_image_path=generated_image_path,
        transform=TransformFashionMNIST.eval,
    )

    return {
        "accuracy": acc,
        "conditioned_accuracy": cond_acc,
        "confusion_matrix": confusion_mat,
    }
