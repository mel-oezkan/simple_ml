from pathlib import Path

import hydra
from omegaconf import DictConfig

from torch.utils.data import Dataset

from vit.evaluation.fid import FID
from vit.models.classifier import (
    CNNFeatureExtractor,
    load_cnn_classifier,
)


def initialize_fid(cfg: DictConfig) -> FID:
    """Build the configured feature extractor and its FID calculator."""
    model_backbone = cfg.eval.get("fid_backbone", "inception")
    if model_backbone == "inception":
        return FID(model_backbone=model_backbone)
    if model_backbone == "fashion-mnist_cnn":
        classifier = load_cnn_classifier(
            Path(cfg.eval.classifier),
            num_classes=cfg.data.n_classes,
        )
        return FID(
            feature_size=classifier.feature_size,
            model_backbone=model_backbone,
            feature_extractor=CNNFeatureExtractor(classifier),
        )
    raise ValueError(f"Unknown FID backbone: {model_backbone!r}")


def eval_fid_test(cfg: DictConfig, real_path: Path, fake_path: Path) -> float:
    """Helper function to run the fid eval and store the results"""
    fid = initialize_fid(cfg)

    return fid.frechet_distance_from_folder(
        folder_real=real_path,
        folder_fake=fake_path,
    )


def eval_fid(cfg: DictConfig, real_ds: Dataset, fake_ds: Dataset) -> float:
    """Helper function to run the fid eval and store the results"""
    fid = initialize_fid(cfg)

    return fid.frechet_from_dataset(
        ds_real=real_ds,
        ds_fake=fake_ds,
        batch_size=cfg.batch_size,
    )


@hydra.main(version_base=None, config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    # todo: check if this is neccessary
    print("Running FID evaluation... not the correct fid")
    eval_fid_test(cfg)


if __name__ == "__main__":
    main()
