from pathlib import Path

from omegaconf import OmegaConf, open_dict

from scripts.eval.datasets import handle_generated_dataset, handle_real_dataset
from scripts.eval.eval_accuracy import eval_classification_accuracy
from scripts.eval.eval_fid import eval_fid
from scripts.eval.generate_eval_samples import GENERATION_CFG_FILE


def eval_model(cfg, current_run_dir: Path) -> dict:
    """Main eval code that computes the FID and conditioned accuracy of a model."""

    # Take the generation settings from the samples, not from the eval config.
    generation_cfg = OmegaConf.load(
        current_run_dir / cfg.eval.generation_id / GENERATION_CFG_FILE
    )
    cfg = cfg.copy()
    with open_dict(cfg):
        cfg.generation = generation_cfg.generation
        cfg.sampler = generation_cfg.sampler

    fake_ds = handle_generated_dataset(cfg, current_run_dir)
    real_ds = handle_real_dataset(cfg)

    return {
        "generation_id": cfg.eval.generation_id,
        "fid": eval_fid(cfg, real_ds, fake_ds),
        "accuracy": eval_classification_accuracy(cfg, fake_ds),
        "generation_config": OmegaConf.to_container(cfg.generation, resolve=True),
        "sampler": OmegaConf.to_container(cfg.sampler, resolve=True),
    }
