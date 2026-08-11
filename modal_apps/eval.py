import json
from pathlib import Path

import modal
from nanoid import generate

from modal_apps.images import PROJECT_ROOT, ml_image
from modal_apps.resources import RUNS_PATH, runs_volume

app = modal.App("diffusion-vit", image=ml_image)

hours = 2
@app.function(
    gpu="L4",
    timeout=hours * 60 * 60,
    volumes={
        RUNS_PATH: runs_volume,
    },
)
def modal_runner(
    overrides: list[str] | None = None
) -> list[tuple[str, bytes]]:
    """Train on an L4 and hand the run artifacts back to the caller."""
    from hydra import compose, initialize
    from scripts.eval import eval_model

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="eval", overrides=overrides or [])

    # extract the run_id from the checkpoint path, which is expected to be in the form
    # /runs/<run_id>/<checkpoint_name>.pt
    configured_checkpoint = Path(cfg.eval.checkpoint_path)
    run_id = configured_checkpoint.parent.name
    
    run_dir = Path("/runs") / run_id
    result: dict = eval_model(cfg, run_dir)
    result_bytes = json.dumps(result, indent=2).encode("utf-8")

    return run_dir, ("result.json", result_bytes)


@app.local_entrypoint()
def cli(overrides: str = ""):
    # e.g. modal run modal_train.py --overrides "epochs=5 batch_size=256"
    run_dir, artifacts = modal_runner.remote(overrides.split() if overrides else [])

    run_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts:
        (run_dir / name).write_bytes(data)

    print(f"Wrote {len(artifacts)} artifacts to {run_dir}")
