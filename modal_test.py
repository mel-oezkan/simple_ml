import subprocess
from pathlib import Path

import modal

project_root = Path(__file__).parent

# todo: rewrite this into the new volume
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-runtime-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install("gcc", "g++", "python3-dev")
    # Installs the locked dependencies from pyproject.toml + uv.lock.
    .uv_sync()
    # Modal 1.x requires local source to be included explicitly.
    .add_local_python_source("vit")
    .add_local_dir(project_root / "tests", remote_path="/root/tests")
    # Mount only the 4.6 MB test fixture, not the entire 2.2 GB data directory.
    .add_local_dir(
        project_root / "data" / "imagenet-debug",
        remote_path="/root/data/imagenet-debug",
    )
)

app = modal.App("simple-ml-tests", image=image)


@app.function(gpu="L4", timeout=60 * 60)
def run_tests() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-ra", "-q", "tests"],
        cwd="/root",
        check=True,
    )


@app.local_entrypoint()
def main() -> None:
    run_tests.remote()