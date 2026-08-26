"""Kaggle-only dataset download, discovery, validation, and preprocessing."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "reliability_class"
EXPECTED_LABELS = {"HIGH", "MEDIUM", "LOW"}

IDENTIFIER_COLUMNS = {
    "circuit_name",
    "source_file",
    "circuit_id",
    "id",
    "index",
}

TARGET_AND_LEAKAGE_COLUMNS = {
    "reliability_class",
    "reliability_score",
    "estimated_fidelity",
    "total_variation_distance",
    "hellinger_distance",
    "success_probability_ideal",
    "success_probability_noisy",
    "ideal_success_probability",
    "noisy_success_probability",
}


class DatasetError(RuntimeError):
    """Raised when the Kaggle dataset cannot be located or validated."""


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    features: pd.DataFrame
    target: pd.Series
    identifiers: pd.DataFrame
    source_path: Path
    sha256: str
    dropped_columns: list[str]


def download_kaggle_dataset(
    slug: str,
    destination: str | Path,
    force: bool = False,
) -> Path:
    """Download a public Kaggle dataset using kagglehub and copy it locally."""
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    if not force:
        try:
            return discover_dataset_csv(destination_path)
        except DatasetError:
            pass

    try:
        import kagglehub
    except ImportError as exc:
        raise DatasetError(
            "kagglehub is not installed. Run START_HERE.bat or pip install -r requirements.txt."
        ) from exc

    try:
        cached_path = Path(kagglehub.dataset_download(slug))
    except Exception as exc:  # kagglehub exposes provider-specific exception types
        raise DatasetError(
            "Kaggle download failed. Download the dataset in a browser and extract it into "
            f"'{destination_path}'. Original error: {exc}"
        ) from exc

    copied: list[str] = []
    for source in cached_path.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(cached_path)
        target = destination_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))

    manifest = {
        "kaggle_slug": slug,
        "kaggle_cache_source": "local KaggleHub cache path omitted",
        "copied_files": sorted(copied),
    }
    (destination_path / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return discover_dataset_csv(destination_path)


def _read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return []


def discover_dataset_csv(directory: str | Path, target_column: str = TARGET) -> Path:
    """Find the largest CSV containing the classification target."""
    root = Path(directory)
    if not root.exists():
        raise DatasetError(f"Dataset directory does not exist: {root}")
    candidates: list[Path] = []
    for path in root.rglob("*.csv"):
        columns = _read_header(path)
        if target_column in columns:
            candidates.append(path)
    if not candidates:
        raise DatasetError(
            f"No CSV containing '{target_column}' was found under '{root}'. "
            "Use download_dataset.py or extract the Kaggle ZIP into data/raw."
        )
    return max(candidates, key=lambda item: item.stat().st_size)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stratified_limit(frame: pd.DataFrame, target_column: str, maximum_rows: int, seed: int) -> pd.DataFrame:
    if maximum_rows <= 0 or len(frame) <= maximum_rows:
        return frame.copy()
    selected, _ = train_test_split(
        frame,
        train_size=maximum_rows,
        random_state=seed,
        stratify=frame[target_column],
    )
    return selected.sort_index().copy()


def load_dataset(
    path: str | Path,
    target_column: str = TARGET,
    maximum_rows: int = 0,
    random_seed: int = 42,
) -> DatasetBundle:
    """Load the QUASAR Kaggle CSV and create a leakage-safe numeric feature matrix."""
    source_path = Path(path)
    try:
        frame = pd.read_csv(source_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read dataset '{source_path}': {exc}") from exc
    if frame.empty:
        raise DatasetError(f"Dataset contains zero rows: {source_path}")
    if target_column not in frame.columns:
        raise DatasetError(f"Required target column '{target_column}' is missing.")

    frame = frame.dropna(subset=[target_column]).copy()
    frame[target_column] = frame[target_column].astype(str).str.strip().str.upper()
    labels = set(frame[target_column].unique())
    missing_labels = EXPECTED_LABELS - labels
    if missing_labels:
        raise DatasetError(
            "Dataset must contain HIGH, MEDIUM, and LOW reliability classes; "
            f"missing {sorted(missing_labels)}. Found {sorted(labels)}."
        )
    frame = frame[frame[target_column].isin(EXPECTED_LABELS)].copy()
    frame = _stratified_limit(frame, target_column, int(maximum_rows), int(random_seed))

    identifier_columns = [column for column in frame.columns if column.lower() in IDENTIFIER_COLUMNS]
    identifiers = pd.DataFrame(index=frame.index)
    if identifier_columns:
        identifiers = frame[identifier_columns].copy()
    if "circuit_name" not in identifiers.columns:
        identifiers["circuit_name"] = [f"circuit_{index}" for index in frame.index]

    drop_lower = IDENTIFIER_COLUMNS | TARGET_AND_LEAKAGE_COLUMNS
    dropped_columns = [column for column in frame.columns if column.lower() in drop_lower]
    candidate_features = frame.drop(columns=dropped_columns, errors="ignore").copy()
    numeric_features = candidate_features.apply(pd.to_numeric, errors="coerce")
    numeric_features = numeric_features.dropna(axis=1, how="all")
    if numeric_features.shape[1] < 5:
        raise DatasetError(
            f"Only {numeric_features.shape[1]} usable numeric features remain after leakage removal."
        )
    constant_columns = [
        column for column in numeric_features.columns if numeric_features[column].nunique(dropna=True) <= 1
    ]
    numeric_features = numeric_features.drop(columns=constant_columns)
    dropped_columns.extend(constant_columns)

    target = frame[target_column].copy()
    numeric_features = numeric_features.loc[target.index]
    identifiers = identifiers.loc[target.index]
    return DatasetBundle(
        frame=frame,
        features=numeric_features,
        target=target,
        identifiers=identifiers,
        source_path=source_path,
        sha256=file_sha256(source_path),
        dropped_columns=sorted(set(dropped_columns)),
    )


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "source_path": str(bundle.source_path),
        "sha256": bundle.sha256,
        "rows": int(len(bundle.frame)),
        "features": int(bundle.features.shape[1]),
        "class_counts": {str(k): int(v) for k, v in bundle.target.value_counts().items()},
        "dropped_columns": bundle.dropped_columns,
        "missing_feature_cells": int(bundle.features.isna().sum().sum()),
    }
