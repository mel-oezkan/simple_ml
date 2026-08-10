from pathlib import Path

import hydra
from omegaconf import DictConfig

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


def eval(cfg: DictConfig):
    """Helper function to run the fid eval and store the results"""
    fid = initialize_fid(cfg)
    if cfg.data.image_dataset:
        fid_distance = fid.frechet_distance_from_folder(
            folder_real=cfg.data.root,
            folder_fake=cfg.data.root,
        )

        print(fid_distance)

    else: 
        raise NotImplementedError("Dataset needs to be a ImageDataset and image_dataset has to be True in the config")

@hydra.main(version_base=None, config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    eval(cfg)


if __name__ == "__main__":
    main()
