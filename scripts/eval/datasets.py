from pathlib import Path

from torch.utils.data import ConcatDataset, Dataset, Subset
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


def _validate_generated_samples(
    cfg,
    dataset_path: Path,
) -> tuple[int, dict[str, int]]:
    """Validate the generated layout and return the requested class size."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Generated dataset not found at {dataset_path}.")

    n_classes = int(cfg.data.n_classes)
    limit = int(cfg.generation.samples)

    # determine the image quanity per class and validate the layout
    expected_class_names = [
        f"class_{class_id}" for class_id in range(n_classes)
    ]
    expected_dirs = set(expected_class_names)
    actual_dirs = {path.name for path in dataset_path.iterdir() if path.is_dir()}
    counts = {
        class_name: len(list((dataset_path / class_name).glob("*.png")))
        for class_name in expected_class_names
    }

    # check for different failure cases
    missing = sorted(expected_dirs - actual_dirs)
    unexpected = sorted(actual_dirs - expected_dirs)
    too_small = any(count < limit for count in counts.values())
    if missing or unexpected or too_small:
        summary = ", ".join(
            f"{class_name}={count:,}" for class_name, count in counts.items()
        )
        raise ValueError(
            f"Generated dataset validation failed at {dataset_path}. "
            f"Expected at least {limit:,} PNG files per class; "
            f"found {sum(counts.values()):,}. Per-class counts: {summary}. "
            f"Missing directories: {missing or 'none'}. "
            f"Unexpected directories: {unexpected or 'none'}."
        )

    # limit can be reached and will be returned
    return limit, counts


def _limit_per_class(
    dataset: ImageFolder,
    limit: int,
    class_counts: dict[str, int],
) -> ConcatDataset:
    """Select the first 'limit' sorted images from every class."""
    subsets = []
    offset = 0

    # ImageFolder stores each class in one contiguous, sorted block.
    for class_name in dataset.classes:
        class_size = class_counts[class_name]
        subsets.append(
            Subset(dataset, range(offset, offset + limit))
        )
        offset += class_size

    return ConcatDataset(subsets)


def handle_generated_dataset(cfg, run_dir: Path) -> Dataset:
    """Load a balanced generated dataset capped at the configured class size."""
    dataset_path = run_dir / cfg.eval.generation_id
    limit, class_counts = _validate_generated_samples(cfg, dataset_path)

    dataset = ImageFolder(
        dataset_path,
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                TransformFashionMNIST.eval,
            ]
        ),
    )
    return _limit_per_class(dataset, limit, class_counts)
