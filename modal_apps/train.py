from pathlib import Path

import modal
from nanoid import generate

from modal_apps.images import PROJECT_ROOT, ml_image
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
        config_path="conf"
    ):
        cfg = compose(config_name="config", overrides=overrides or [])

    run_dir = Path(RUNS_PATH) / run_id
    train(cfg, run_dir, runs_volume.commit)

    return [(p.name, p.read_bytes()) for p in sorted(run_dir.iterdir()) if p.is_file()]


def save_artifacts(run_id: str, artifacts: list[tuple[str, bytes]]) -> None:
    """Helper function to store the run artifacts in modal.

    Args:
        run_id (str): id of the run
        artifacts (list[tuple[str, bytes]]): genereated artifacts that will be store.
    """
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # write to modal
    for name, data in artifacts:
        (run_dir / name).write_bytes(data)

    print(f"Wrote {len(artifacts)} artifacts to {run_dir}")


@app.local_entrypoint()
def cli(overrides: str = "", sweep: str = "", gpu: str = "L4"):
    # single run: modal run modal_apps/train.py --overrides "epochs=5 batch_size=256"
    # parallel:   modal run modal_apps/train.py --gpu H100 \
    #                 --sweep "learning_rate=1e-4; learning_rate=3e-4"
    # every ";"-separated override set becomes its own run on its own GPU
    train_fn = modal_train.with_options(gpu=gpu)
    override_sets = [s.split() for s in sweep.split(";")] if sweep else [overrides.split()]

    calls = []
    for run_overrides in override_sets:
        run_id = generate()
        print(f"Run ID: {run_id} <- {' '.join(run_overrides) or '(defaults)'}")
        calls.append((run_id, train_fn.spawn(run_id, run_overrides)))

    for run_id, call in calls:
        save_artifacts(run_id, call.get())
