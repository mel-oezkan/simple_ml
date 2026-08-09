import modal

RUNS_PATH = "/runs"
DATASETS_PATH = "/datasets"
GENERATIONS_PATH = "/generations"

runs_volume = modal.Volume.from_name(
    "diffusion-runs",
    create_if_missing=True,
)

datasets_volume = modal.Volume.from_name(
    "simple-ml-datasets",
    create_if_missing=True,
)

generations_volume = modal.Volume.from_name(
    "diffusion-generations",
    create_if_missing=True,
)