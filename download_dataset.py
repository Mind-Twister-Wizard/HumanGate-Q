"""Download the selected public Kaggle dataset into data/raw."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from humangateq.configuration import load_config, resolve_project_path
from humangateq.data import DatasetError, download_kaggle_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    destination = resolve_project_path(config["dataset"]["raw_directory"], PROJECT_ROOT)
    try:
        dataset_path = download_kaggle_dataset(
            config["dataset"]["kaggle_slug"], destination, force=args.force
        )
    except DatasetError as exc:
        print(f"ERROR: {exc}")
        return 2
    print("Kaggle dataset ready.")
    print(f"Selected CSV: {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

