from pathlib import Path

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

from vit.data.transforms import TransformFashionMNIST


def handle_real_dataset(cfg) -> Dataset:
    """Load the supported real evaluation dataset."""
    if cfg.eval.real_dataset != "data/fashion-mnist":
        raise NotImplementedError(
            "Eval only supports Fashion-MNIST at 'data/fashion-mnist'."
        )

    dataset_path = Path(cfg.eval.real_dataset)
    return FashionMNIST(
        root=dataset_path,
        download=not dataset_path.exists(),
        transform=TransformFashionMNIST.eval,
        train=False,
    )


def _validate_generated_samples(cfg, dataset_path: Path) -> int:
    """Validate the generated layout and return the requested class size."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Generated dataset not found at {dataset_path}.")

    n_classes = int(cfg.data.n_classes)
    limit = int(cfg.generation.samples)
    expected_dirs = {f"class_{class_id}" for class_id in range(n_classes)}
    actual_dirs = {path.name for path in dataset_path.iterdir() if path.is_dir()}
    counts = {
        class_id: len(list((dataset_path / f"class_{class_id}").glob("*.png")))
        for class_id in range(n_classes)
    }

    missing = sorted(expected_dirs - actual_dirs)
    unexpected = sorted(actual_dirs - expected_dirs)
    too_small = any(count < limit for count in counts.values())
    if missing or unexpected or too_small:
        summary = ", ".join(
            f"class_{class_id}={count:,}" for class_id, count in counts.items()
        )
        raise ValueError(
            f"Generated dataset validation failed at {dataset_path}. "
            f"Expected at least {limit:,} PNG files per class; "
            f"found {sum(counts.values()):,}. Per-class counts: {summary}. "
            f"Missing directories: {missing or 'none'}. "
            f"Unexpected directories: {unexpected or 'none'}."
        )

    return limit


def _limit_per_class(dataset: ImageFolder, limit: int) -> Subset:
    """Select the first ``limit`` sorted images from every class."""
    selected = []
    counts = [0] * len(dataset.classes)
    for index, class_id in enumerate(dataset.targets):
        if counts[class_id] < limit:
            selected.append(index)
            counts[class_id] += 1
    return Subset(dataset, selected)


def handle_generated_dataset(cfg, run_dir: Path) -> Dataset:
    """Load a balanced generated dataset capped at the configured class size."""
    dataset_path = run_dir / cfg.eval.generation_id
    limit = _validate_generated_samples(cfg, dataset_path)
    dataset = ImageFolder(
        dataset_path,
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                TransformFashionMNIST.eval,
            ]
        ),
    )
    return _limit_per_class(dataset, limit)
