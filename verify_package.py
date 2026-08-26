"""Verify dependencies, required files, tests, and an optional downloaded dataset."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humangateq_matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


REQUIRED_IMPORTS = (
    "joblib",
    "kagglehub",
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "sklearn",
    "yaml",
)

REQUIRED_FILES = (
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "config.yaml",
    "run_all.py",
    "compare_to_paper_run.py",
    "download_dataset.py",
    "src/humangateq/pipeline.py",
    "src/humangateq/policies.py",
    "docs/METHODOLOGY.md",
    "results/paper_run/run_metadata.json",
)

EXPECTED_PAPER_DATASET_SHA256 = (
    "506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c"
)


def main() -> int:
    print("HumanGate-Q package verification")
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10 or later is required.")
        return 1

    missing_imports = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[OK] dependency: {module_name}")
        except ImportError:
            missing_imports.append(module_name)
            print(f"[MISSING] dependency: {module_name}")
    if missing_imports:
        print("Run START_HERE.bat to install missing packages.")
        return 1

    missing_files = [name for name in REQUIRED_FILES if not (PROJECT_ROOT / name).exists()]
    if missing_files:
        print(f"ERROR: Package files missing: {missing_files}")
        return 1
    print(f"[OK] {len(REQUIRED_FILES)} required package files")

    print("Running automated tests...")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    )
    if result.returncode:
        print("ERROR: Automated tests failed.")
        return result.returncode

    from humangateq.data import (
        DatasetError,
        dataset_summary,
        discover_dataset_csv,
        file_sha256,
        load_dataset,
    )

    try:
        path = discover_dataset_csv(PROJECT_ROOT / "data" / "raw")
    except DatasetError:
        print("[INFO] Kaggle data not present yet; source-code tests passed using an in-memory fixture.")
    else:
        digest = file_sha256(path)
        if digest == EXPECTED_PAPER_DATASET_SHA256:
            print("[OK] dataset SHA-256 matches the archived chapter run")
        else:
            print(
                "[WARNING] dataset SHA-256 differs from the archived chapter run; "
                "the code can run, but exact numerical reproduction is not guaranteed."
            )
        try:
            data = load_dataset(path, maximum_rows=1000, random_seed=42)
        except DatasetError as exc:
            print(f"ERROR: Downloaded dataset validation failed: {exc}")
            return 1
        summary = dataset_summary(data)
        print(
            f"[OK] Kaggle schema: {summary['rows']} checked rows, "
            f"{summary['features']} leakage-safe features"
        )

    print("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
