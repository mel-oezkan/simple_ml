from pathlib import Path

from omegaconf import OmegaConf

from scripts.eval.datasets import handle_generated_dataset, handle_real_dataset
from scripts.eval.eval_accuracy import eval_classification_accuracy
from scripts.eval.eval_fid import eval_fid


def eval_model(cfg, current_run_dir: Path) -> dict:
    """Main eval code that computes the FID and conditioned accuracy of a model."""

    fake_ds = handle_generated_dataset(cfg, current_run_dir)
    real_ds = handle_real_dataset(cfg)

    return {
        "generation_id": cfg.eval.generation_id,
        "fid": eval_fid(cfg, real_ds, fake_ds),
        "accuracy": eval_classification_accuracy(cfg, fake_ds),
        "generation_config": OmegaConf.to_container(cfg.generation, resolve=True),
    }
