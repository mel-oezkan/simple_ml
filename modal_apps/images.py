from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REMOTE_CONFIG_PATH = "/root/conf"

CUDA_TAG = "13.0.2-runtime-ubuntu24.04"


base_image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install("gcc", "g++", "python3-dev")
    .pip_install("torch>=2.13.0", "torchvision>=0.28.0")
    .pip_install("einops", "hydra-core", "matplotlib", "nanoid", "tqdm", "scipy")
 
)

ml_image = (
    base_image
    .add_local_python_source("modal_apps", "vit", "scripts")
    # mirror the local layout so `from vit.diffusion import ...` and hydra's
    # relative config_path both resolve the same way they do on a laptop
    .add_local_dir(PROJECT_ROOT / "conf", remote_path=REMOTE_CONFIG_PATH)
    .add_local_file(PROJECT_ROOT / "main.py", remote_path="/root/main.py")   
) 
