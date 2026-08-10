import argparse
import json
import sys
from pathlib import Path

# When this file is run directly, Python only adds ``scripts/visualize`` to
# sys.path. Add the repository root so the local ``vit`` package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vit.visualize.training_visualizations import plot_loss


def main():
    parser = argparse.ArgumentParser(description="Plot training losses from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing training losses.")
    args = parser.parse_args()

    json_file = Path(args.json_file)
    with json_file.open() as f:
        losses = json.load(f)

    save_path = json_file.with_suffix(".png")
    plot_loss(save_path, losses)


if __name__ == "__main__":
    main()