import shutil
import urllib.request
from pathlib import Path

import modal

dataset_volume = modal.Volume.from_name(
    "simple-ml-datasets",
    create_if_missing=True,
)