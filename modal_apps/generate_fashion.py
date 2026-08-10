import random
import shlex
from pathlib import Path
from uuid import UUID, uuid4

import modal

from modal_apps.images import PROJECT_ROOT, ml_image
from modal_apps.resources import (
    GENERATIONS_PATH,
    RUNS_PATH,
    generations_volume,
    runs_volume,
)
from scripts.eval.generate_eval_samples import generate_samples
from vit.utils.random import set_seed

app = modal.App("diffusion-vit", image=ml_image)

hours = 2


@app.function(
    gpu="L4",
    timeout=hours * 60 * 60,
    volumes={
        RUNS_PATH: runs_volume,
        GENERATIONS_PATH: generations_volume,
    },
)
def modal_runner(
    generation_id: str,
    overrides: list[str] | None = None,
) -> tuple[str, list[tuple[str, bytes]]]:
    """Generate samples into a Volume and return its path and a preview."""
    from hydra import compose, initialize
    from omegaconf import OmegaConf

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="eval", overrides=overrides or [])

    set_seed(cfg.seed)

    preview_count = cfg.eval.preview_count
    if preview_count < 0:
        raise ValueError("preview_count must be non-negative")

    configured_checkpoint = Path(cfg.eval.checkpoint_path)
    run_id = str(UUID(configured_checkpoint.parent.name))
    generation_id = str(UUID(generation_id))

    checkpoint_path = Path(RUNS_PATH) / run_id / configured_checkpoint.name
    cfg.eval.checkpoint_path = str(checkpoint_path)
    OmegaConf.update(
        cfg,
        "training.checkpoint_path",
        str(checkpoint_path),
        force_add=True,
    )

    save_dir = Path(GENERATIONS_PATH) / run_id / generation_id
    generate_samples(cfg, save_dir)

    image_paths = sorted(save_dir.rglob("*.png"))
    rng = random.Random(cfg.get("seed", None))
    preview_paths = rng.sample(
        image_paths,
        k=min(preview_count, len(image_paths)),
    )
    previews = [
        (str(path.relative_to(save_dir)), path.read_bytes())
        for path in preview_paths
    ]

    # Make every generated image visible through the Volume after this call.
    generations_volume.commit()
    volume_path = str(save_dir.relative_to(GENERATIONS_PATH))
    return volume_path, previews


@app.local_entrypoint()
def cli(overrides: str = ""):
    generation_id = str(uuid4())
    volume_path, previews = modal_runner.remote(
        generation_id,
        shlex.split(overrides),
    )

    preview_dir = PROJECT_ROOT / "runs" / volume_path
    for relative_path, data in previews:
        local_path = preview_dir / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(data)

    print(f"Generated images in diffusion-generations/{volume_path}")
    print(f"Wrote {len(previews)} preview images to {preview_dir}")
