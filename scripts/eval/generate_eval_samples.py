from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from vit.model_utils import load_checkpoint

# Written next to the samples; eval reads the generation settings from it.
GENERATION_CFG_FILE = "generation_cfg.yaml"


def save_batch(samples: torch.Tensor, class_dir: Path, batch_index: int) -> None:
    """Encode and save one generated batch from a background thread."""
    for sample_index, sample in enumerate(samples):
        image_path = class_dir / f"{batch_index:05d}_{sample_index:05d}.png"
        save_image(
            sample,
            image_path,
            normalize=True,
            value_range=(-1, 1),
        )


def generate_samples(cfg, save_dir: Path | None = None) -> None:
    """Generate the samples from the trained model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = Path(save_dir or cfg.generation.get("save_dir", "eval_samples"))

    max_pending_batches = cfg.generation.get("max_pending_batches", 4)
    save_workers = cfg.generation.get("save_workers", 2)

    model, ema, _ = load_checkpoint(cfg, device)
    model.eval()

    sampler = hydra.utils.instantiate(cfg.sampler, model=model)

    # handle the generation and saving via threds
    # TODO: go though this code again
    pending: deque[Future[None]] = deque()
    with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        with ema.averaged(model):
            for class_index in tqdm(
                range(cfg.data.n_classes), desc="Generating samples"
            ):
                class_dir = save_dir / f"class_{class_index}"
                class_dir.mkdir(parents=True, exist_ok=True)

                samples_per_class = cfg.generation.samples
                batch_size = cfg.generation.batch_size

                for batch_start in tqdm(
                    range(0, samples_per_class, batch_size),
                    desc=f"Generating samples for class {class_index}",
                ):
                    current_batch_size = min(
                        batch_size,
                        samples_per_class - batch_start,
                    )

                    labels = torch.full(
                        (current_batch_size,),
                        class_index,
                        dtype=torch.long,
                        device=device,
                    )

                    denoised_samples = sampler.sample(
                        n_samples=labels.shape[0],
                        device=torch.device(device),
                        y=labels,
                        guidance_scale=cfg.generation.get("guidance", 1.0),
                    )

                    # Copy before queueing so the worker never touches CUDA state.
                    cpu_samples = denoised_samples.detach().cpu()
                    pending.append(
                        save_pool.submit(
                            save_batch,
                            cpu_samples,
                            class_dir,
                            batch_start // batch_size,
                        )
                    )

                    # Apply backpressure instead of allowing queued images to use
                    # unbounded host memory when the disk cannot keep up.
                    if len(pending) >= max_pending_batches:
                        pending.popleft().result()

            # Wait for the final writes and surface any exception from a worker.
            while pending:
                pending.popleft().result()

    # Eval reads the settings from here, so results stay tied to the samples.
    OmegaConf.save(cfg, save_dir / GENERATION_CFG_FILE, resolve=True)


@hydra.main(version_base=None, config_path="../../conf", config_name="generate")
def main(cfg: DictConfig, save_dir: Path | None = None) -> None:
    """Generate the samples from the trained model."""
    generate_samples(cfg, save_dir)


if __name__ == "__main__":
    main()
