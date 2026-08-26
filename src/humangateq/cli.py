"""Command-line entry point for HumanGate-Q."""

from __future__ import annotations

import argparse
from pathlib import Path

from .configuration import load_config, make_quick_config
from .data import DatasetError
from .pipeline import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-path")
    parser.add_argument("--output")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--download-if-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path.cwd()
    config = load_config(project_root / args.config)
    if args.quick:
        config = make_quick_config(config)
    try:
        result = run_experiment(
            config,
            project_root=project_root,
            data_path=args.data_path,
            output_directory=args.output,
            download_if_missing=args.download_if_missing,
        )
    except DatasetError as exc:
        print(f"DATASET ERROR: {exc}")
        return 2
    print()
    print("HumanGate-Q experiment completed successfully.")
    print(f"Runtime: {result['runtime_seconds']:.2f} seconds")
    print(f"Results: {result['output_directory']}")
    print(f"Summary: {result['artifacts']['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

