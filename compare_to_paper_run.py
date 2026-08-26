"""Compare a fresh HumanGate-Q run with the immutable chapter evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent


def _read_table(root: Path, filename: str, index_column: str | int) -> pd.DataFrame:
    frame = pd.read_csv(root / "tables" / filename)
    selected = frame.columns[index_column] if isinstance(index_column, int) else index_column
    return frame.set_index(selected)


def _compare_numeric(
    label: str,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    excluded: set[str] | None = None,
) -> tuple[bool, float]:
    excluded = excluded or set()
    if set(reference.index) != set(candidate.index):
        print(f"[FAIL] {label}: row labels differ")
        return False, float("inf")
    common = [
        column
        for column in reference.columns
        if column in candidate.columns
        and column not in excluded
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(candidate[column])
    ]
    left = reference.loc[candidate.index, common].to_numpy(dtype=float)
    right = candidate[common].to_numpy(dtype=float)
    difference = float(np.nanmax(np.abs(left - right))) if common else 0.0
    passed = bool(np.allclose(left, right, rtol=1e-9, atol=1e-12, equal_nan=True))
    print(f"[{'OK' if passed else 'FAIL'}] {label}: max absolute difference {difference:.3g}")
    return passed, difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "latest",
        help="Fresh run directory to compare (default: outputs/latest).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "results" / "paper_run",
        help="Archived run directory (default: results/paper_run).",
    )
    args = parser.parse_args()
    reference = args.reference.resolve()
    candidate = args.candidate.resolve()
    if not reference.is_dir() or not candidate.is_dir():
        print("ERROR: both reference and candidate run directories must exist")
        return 2

    checks: list[bool] = []
    table_specs = (
        ("model metrics", "model_metrics.csv", 0, set()),
        ("model comparison", "model_comparison.csv", 0, set()),
        ("policy metrics", "policy_metrics.csv", 0, set()),
        ("expanded ablation", "ablation_metrics.csv", 0, set()),
        (
            "model selection",
            "model_selection_cv.csv",
            "candidate",
            {"cv_fit_seconds_mean", "search_seconds"},
        ),
    )
    for label, filename, index_column, excluded in table_specs:
        passed, _ = _compare_numeric(
            label,
            _read_table(reference, filename, index_column),
            _read_table(candidate, filename, index_column),
            excluded=excluded,
        )
        checks.append(passed)

    reference_metadata = json.loads(
        (reference / "run_metadata.json").read_text(encoding="utf-8")
    )
    candidate_metadata = json.loads(
        (candidate / "run_metadata.json").read_text(encoding="utf-8")
    )
    metadata_match = (
        reference_metadata["dataset"]["sha256"]
        == candidate_metadata["dataset"]["sha256"]
        and reference_metadata["selected_candidate"]
        == candidate_metadata["selected_candidate"]
        and reference_metadata["configuration"]["risk"]["thresholds"]
        == candidate_metadata["configuration"]["risk"]["thresholds"]
    )
    print(f"[{'OK' if metadata_match else 'FAIL'}] data hash, model, and thresholds")
    checks.append(metadata_match)

    reference_actions = pd.read_csv(
        reference / "tables" / "workflow_decisions.csv",
        usecols=["action__HumanGate-Q"],
    )
    candidate_actions = pd.read_csv(
        candidate / "tables" / "workflow_decisions.csv",
        usecols=["action__HumanGate-Q"],
    )
    action_match = reference_actions.equals(candidate_actions)
    print(f"[{'OK' if action_match else 'FAIL'}] ordered HumanGate-Q decisions")
    checks.append(action_match)

    if all(checks):
        print("VERIFICATION PASSED: the fresh run reproduces the chapter evidence.")
        return 0
    print("VERIFICATION FAILED: inspect the differences above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
