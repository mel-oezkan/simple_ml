import uuid
from pathlib import Path

import modal

from modal_apps.images import PROJECT_ROOT, REMOTE_CONFIG_PATH, ml_image
from modal_apps.resources import RUNS_PATH, runs_volume



app = modal.App("diffusion-vit", image=ml_image)


@app.function(gpu="L4", timeout=24 * 60 * 60, volumes={RUNS_PATH: runs_volume})
def modal_train(
    run_id: str, overrides: list[str] | None = None
) -> list[tuple[str, bytes]]:
    """Train on an L4 and hand the run artifacts back to the caller."""
    from hydra import compose, initialize
    from main import train

    with initialize(
        version_base=None, 
        config_path=REMOTE_CONFIG_PATH
    ):
        cfg = compose(config_name="config", overrides=overrides or [])

    run_dir = Path("/runs") / run_id
    train(cfg, run_dir, runs_volume.commit)

    return [(p.name, p.read_bytes()) for p in sorted(run_dir.iterdir()) if p.is_file()]


@app.local_entrypoint()
def cli(overrides: str = ""):
    # e.g. modal run modal_train.py --overrides "epochs=5 batch_size=256"
    run_id = str(uuid.uuid4())
    print(f"Run ID: {run_id}")
    artifacts = modal_train.remote(
        run_id, 
        overrides.split() if overrides else []
    )

    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    for name, data in artifacts:
        (run_dir / name).write_bytes(data)

    print(f"Wrote {len(artifacts)} artifacts to {run_dir}")
