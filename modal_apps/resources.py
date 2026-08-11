import modal

RUNS_PATH = "/runs"
DATASETS_PATH = "/datasets"

runs_volume = modal.Volume.from_name(
    "diffusion-runs",
    create_if_missing=True,
)

datasets_volume = modal.Volume.from_name(
    "simple-ml-datasets",
    create_if_missing=True,
)
