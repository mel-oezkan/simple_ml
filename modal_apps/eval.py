from pathlib import Path

import modal
from nanoid import generate

# torch 2.13 resolves nvidia-cuda-runtime 13.x, so the base image must be CUDA 13.
# `runtime` (not `devel`) is enough for the torch wheels, which bundle their CUDA
# libs, but Triton compiles a small C extension at runtime and needs a host
# compiler, so gcc/python headers are installed below.
cuda_version = "13.0.2"
flavor = "runtime"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

project_root = Path(__file__).parent

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install("gcc", "g++", "python3-dev")
    .pip_install("torch>=2.13.0", "torchvision>=0.28.0")
    .pip_install("einops", "hydra-core", "matplotlib", "nanoid", "tqdm")
    # mirror the local layout so `from vit.diffusion import ...` and hydra's
    # relative config_path both resolve the same way they do on a laptop
    .add_local_dir(project_root / "vit", remote_path="/root/vit")
    .add_local_dir(project_root / "conf", remote_path="/root/conf")
    .add_local_file(project_root / "main.py", remote_path="/root/main.py")
)



app = modal.App("diffusion-vit", image=image)

volume = modal.Volume.from_name("diffusion-runs", create_if_missing=True)


@app.function(gpu="L4", timeout=24 * 60 * 60, volumes={"/runs": volume})
def modal_runner(
    run_id: str, overrides: list[str] | None = None
) -> list[tuple[str, bytes]]:
    """Train on an L4 and hand the run artifacts back to the caller."""
    from hydra import compose, initialize
    from main import train

    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=overrides or [])

    run_dir = Path("/runs") / run_id
    train(cfg, run_dir, volume.commit)

    return [(p.name, p.read_bytes()) for p in sorted(run_dir.iterdir()) if p.is_file()]


@app.local_entrypoint()
def cli(overrides: str = ""):
    # e.g. modal run modal_train.py --overrides "epochs=5 batch_size=256"
    run_id = generate()
    print(f"Run ID: {run_id}")
    artifacts = modal_runner.remote(run_id, overrides.split() if overrides else [])

    run_dir = project_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts:
        (run_dir / name).write_bytes(data)

    print(f"Wrote {len(artifacts)} artifacts to {run_dir}")
