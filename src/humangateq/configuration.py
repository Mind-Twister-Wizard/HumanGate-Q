"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration is missing or inconsistent."""


REQUIRED_PATHS = (
    ("project", "random_seed"),
    ("dataset", "kaggle_slug"),
    ("dataset", "raw_directory"),
    ("dataset", "target_column"),
    ("model", "n_estimators"),
    ("scenarios", "proportions"),
    ("risk", "weights"),
    ("risk", "thresholds"),
    ("output", "directory"),
)


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML configuration and perform lightweight validation."""
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Could not read {config_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError("The configuration root must be a mapping.")

    for path_parts in REQUIRED_PATHS:
        current: Any = loaded
        for part in path_parts:
            if not isinstance(current, dict) or part not in current:
                raise ConfigurationError(f"Missing configuration key: {'.'.join(path_parts)}")
            current = current[part]

    proportions = loaded["scenarios"]["proportions"]
    if not isinstance(proportions, dict) or not proportions:
        raise ConfigurationError("scenarios.proportions must be a non-empty mapping.")
    proportion_sum = sum(float(value) for value in proportions.values())
    if abs(proportion_sum - 1.0) > 1e-6:
        raise ConfigurationError(
            f"Scenario proportions must sum to 1.0; received {proportion_sum:.6f}."
        )

    thresholds = loaded["risk"]["thresholds"]
    ordered = [float(thresholds[name]) for name in ("repair", "human_review", "abstain")]
    if ordered != sorted(ordered) or not all(0.0 <= value <= 1.0 for value in ordered):
        raise ConfigurationError("Risk thresholds must satisfy 0 <= repair <= review <= abstain <= 1.")
    return loaded


def make_quick_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a fast, scientifically identical configuration for validation runs."""
    quick = deepcopy(config)
    quick["dataset"]["maximum_rows"] = min(int(quick["dataset"]["maximum_rows"]), 6000)
    quick["model"]["n_estimators"] = min(int(quick["model"]["n_estimators"]), 60)
    quick["model"]["max_depth"] = min(int(quick["model"].get("max_depth", 16)), 16)
    quick["model"]["validation_folds"] = 2
    if isinstance(quick["model"].get("candidates"), list):
        quick["model"]["candidates"] = quick["model"]["candidates"][:2]
        for candidate in quick["model"]["candidates"]:
            candidate["n_estimators"] = min(int(candidate.get("n_estimators", 80)), 80)
    quick["scenarios"]["maximum_test_cases"] = min(
        int(quick["scenarios"]["maximum_test_cases"]), 1200
    )
    quick["evaluation"]["bootstrap_repetitions"] = min(
        int(quick["evaluation"]["bootstrap_repetitions"]), 50
    )
    quick["output"]["directory"] = "outputs/quick"
    return quick


def resolve_project_path(value: str | Path, project_root: str | Path) -> Path:
    """Resolve a config path relative to the package root."""
    candidate = Path(value)
    return candidate if candidate.is_absolute() else Path(project_root) / candidate
