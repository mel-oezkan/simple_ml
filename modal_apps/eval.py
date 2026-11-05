import json
from pathlib import Path

import modal

from modal_apps.images import PROJECT_ROOT, ml_image
from modal_apps.resources import RUNS_PATH, runs_volume

app = modal.App("diffusion-vit", image=ml_image)

EVAL_RESULT_FILE = "eval_result.json"

hours = 2
@app.function(
    gpu="L4",
    timeout=hours * 60 * 60,
    volumes={
        RUNS_PATH: runs_volume,
    },
)
def modal_runner(
    cfg_overrides: list[str] | None = None
) -> str:
    """Evaluate a run and persist its result in the runs volume."""
    from hydra import compose, initialize
    from scripts.eval import eval_model

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="eval", overrides=cfg_overrides or [])

    # handle the problem of defining path as runs/ instead of /runs/
    for key in ("classifier", "accuracy_classifier"):
        configured_classifier = Path(cfg.eval[key])
        cfg.eval[key] = str(
            Path(RUNS_PATH) / configured_classifier.relative_to("runs")
        )

    # extract the run_id from the checkpoint path, which is expected to be in the form
    # /runs/<run_id>/<checkpoint_name>.pt
    configured_checkpoint = Path(cfg.eval.checkpoint_path)
    run_id = configured_checkpoint.parent.name
    
    run_dir = Path(RUNS_PATH) / run_id
    new_result: dict = eval_model(cfg, run_dir)

    # Store the result next to the samples it scores, one file per generation.
    result_path = run_dir / cfg.eval.generation_id / EVAL_RESULT_FILE
    result_path.write_text(json.dumps(new_result, indent=2), encoding="utf-8")
    runs_volume.commit()

    return result_path.relative_to(RUNS_PATH).as_posix()


@app.local_entrypoint()
def cli(overrides: str = ""):
    result_path = modal_runner.remote(overrides.split() if overrides else [])

    local_path = PROJECT_ROOT / "runs" / result_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as result_file:
        for chunk in runs_volume.read_file(result_path):
            result_file.write(chunk)

    print(f"Wrote result to diffusion-runs/{result_path}")
    print(f"Downloaded result to {local_path}")
