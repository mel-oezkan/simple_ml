from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torchvision.utils import save_image
from tqdm import tqdm

from vit.model_utils import load_checkpoint


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
    save_dir = Path(save_dir or cfg.eval.get("save_dir", "eval_samples"))

    max_pending_batches = cfg.eval.get("max_pending_batches", 4)
    save_workers = cfg.eval.get("save_workers", 2)

    model, ema, _ = load_checkpoint(cfg, device)
    model.eval()

    pending: deque[Future[None]] = deque()
    with ThreadPoolExecutor(max_workers=save_workers) as save_pool:
        with ema.averaged(model):
            for class_index in tqdm(
                range(cfg.data.n_classes), desc="Generating samples"
            ):
                class_dir = save_dir / f"class_{class_index}"
                class_dir.mkdir(parents=True, exist_ok=True)

                for batch_index in tqdm(
                    range(
                        0, 
                        cfg.generation.samples, 
                        cfg.generation.batch_size
                    ),
                    desc=f"Generating samples for class {class_index}",
                ):
                    labels = torch.full(
                        (cfg.generation.batch_size,),
                        class_index,
                        dtype=torch.long,
                        device=device,
                    )

                    denoised_samples = model.sample(
                        n=labels.shape[0],
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
                            batch_index,
                        )
                    )

                    # Apply backpressure instead of allowing queued images to use
                    # unbounded host memory when the disk cannot keep up.
                    if len(pending) >= max_pending_batches:
                        pending.popleft().result()

            # Wait for the final writes and surface any exception from a worker.
            while pending:
                pending.popleft().result()


@hydra.main(version_base=None, config_path="../../conf", config_name="eval")
def main(cfg: DictConfig, save_dir: Path | None = None) -> None:
    """Generate the samples from the trained model."""
    generate_samples(cfg, save_dir)


if __name__ == "__main__":
    main()
