import logging
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import FashionMNIST, ImageFolder

from scripts.eval.eval_accuracy import eval_classification_accuracy
from scripts.eval.eval_fid import eval_fid
from vit.data.transforms import TransformFashionMNIST


def handle_real_dataset(cfg) -> Path:
    """Helper funciton to check if the mnist dataset exists"""
    if cfg.eval.real_dataset == "data/fashion-mnist":
        # check if dataset exists and if not download
        ds_path = Path(cfg.eval.real_dataset)

        dowload = not ds_path.exists()
        ds = FashionMNIST(
            root=ds_path,
            download=dowload,
            transform=TransformFashionMNIST.eval,
            train=False,
        )

    else:
        raise NotImplementedError(
            "Eval currently only supports Fashion-MNIST as "
            "the real dataset. Please set cfg.eval.real_dataset "
            "to 'data/fashion-mnist'."
        )

    return ds


def handle_generated_dataset(cfg, run_dir: Path) -> Path:
    """Helper function to make loading generated datasets check more gracefully.

    Would also be bad manners to implement something for real_dataset but
    not the fake one so here we go.
    """

    generated_data_path = run_dir / cfg.eval.generation_id
    if not generated_data_path.exists():
        raise FileNotFoundError(
            f"Generated dataset not found at {generated_data_path}. "
            "Please run the generation step first."
        )

    return ImageFolder(
        generated_data_path,
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                TransformFashionMNIST.eval,
            ]
        ),
    )


def eval_model(cfg, current_run_dir: Path) -> dict:
    """Main eval code that computes the FID and conditioned accuracy of a model."""

    # check if the generated dataset is 10k samples (1k per class)
    dataset_path = current_run_dir / cfg.eval.generation_id
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Generated dataset not found at {dataset_path}. "
            "Please run the generation step first."
        )

    dataset_count = {class_id: 0 for class_id in range(cfg.data.n_classes)}
    for class_dir in dataset_path.iterdir():
        # soft per class check
        if class_dir.is_dir():
            class_count = len(list(class_dir.glob("*.png")))

            # image is saved as class_{class_id}
            class_id = int(class_dir.name.split("_")[-1])
            dataset_count[class_id] = class_count
            if class_count <= 1000:
                logging.warning(
                    f"Class {class_id} has only {class_count} samples. "
                    "Please run the generation step with the correct number of samples."
                )

    if sum(dataset_count.values()) != 10000:
        raise ValueError(
            f"Generated dataset at {dataset_path} does not contain 10k samples. "
            "Please run the generation step with the correct number of samples."
        )

    # define the real dataset path and the generated dataset path
    real_ds = handle_real_dataset(cfg)
    fake_ds = handle_generated_dataset(cfg, current_run_dir)

    results = {}
    results["fid"] = eval_fid(cfg, real_ds, fake_ds)
    results["accuracy"] = eval_classification_accuracy(
        cfg, current_run_dir, cfg.eval.generation_id
    )

    return results
