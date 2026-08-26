"""Controlled high-stakes workflow scenarios derived only from held-out Kaggle rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .modeling import COMPLEXITY_COLUMNS, ReliabilityModelBundle

DOMAIN_SPECS: dict[str, tuple[float, float, float]] = {
    "health_research": (0.94, 0.025, 0.16),
    "cryptographic_analysis": (0.91, 0.025, 0.16),
    "financial_optimization": (0.84, 0.035, 0.17),
    "materials_discovery": (0.76, 0.045, 0.18),
    "logistics_optimization": (0.66, 0.050, 0.18),
    "education_demo": (0.30, 0.050, 0.15),
}


@dataclass
class ScenarioBundle:
    features: pd.DataFrame
    metadata: pd.DataFrame


def _limited_test_set(
    features: pd.DataFrame,
    target: pd.Series,
    maximum_cases: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if maximum_cases <= 0 or len(features) <= maximum_cases:
        return features.copy(), target.copy()
    selected, _, y_selected, _ = train_test_split(
        features,
        target,
        train_size=maximum_cases,
        random_state=seed,
        stratify=target,
    )
    return selected.sort_index(), y_selected.loc[selected.sort_index().index]


def build_scenarios(
    model_bundle: ReliabilityModelBundle,
    test_features: pd.DataFrame,
    test_target: pd.Series,
    identifiers: pd.DataFrame,
    config: dict[str, Any],
    random_seed: int,
) -> ScenarioBundle:
    """Assign reproducible contexts and inject controlled test-time failures."""
    maximum_cases = int(config.get("maximum_test_cases", len(test_features)))
    features, target = _limited_test_set(test_features, test_target, maximum_cases, random_seed)
    identifiers = identifiers.reindex(features.index)
    rng = np.random.default_rng(random_seed + 17)

    scenario_names = list(config["proportions"].keys())
    scenario_probabilities = np.array(list(config["proportions"].values()), dtype=float)
    scenario_probabilities /= scenario_probabilities.sum()
    scenarios = rng.choice(scenario_names, size=len(features), p=scenario_probabilities)

    domains = list(DOMAIN_SPECS.keys())
    domain_probabilities = np.array([DOMAIN_SPECS[name][2] for name in domains], dtype=float)
    domain_probabilities /= domain_probabilities.sum()
    assigned_domains = rng.choice(domains, size=len(features), p=domain_probabilities)
    criticality = np.array(
        [
            np.clip(rng.normal(DOMAIN_SPECS[name][0], DOMAIN_SPECS[name][1]), 0.0, 1.0)
            for name in assigned_domains
        ]
    )

    perturbed = features.copy()
    metadata = pd.DataFrame(index=features.index)
    metadata["circuit_id"] = identifiers.get(
        "circuit_name", pd.Series([f"circuit_{idx}" for idx in features.index], index=features.index)
    ).astype(str)
    metadata["true_reliability"] = target.astype(str)
    metadata["scenario"] = scenarios
    metadata["workflow_domain"] = assigned_domains
    metadata["workflow_criticality"] = criticality
    metadata["ambiguous_goal"] = (metadata["scenario"] == "ambiguous_goal").astype(int)
    metadata["tool_failure"] = (metadata["scenario"] == "tool_failure").astype(int)

    missing_rows = metadata.index[metadata["scenario"] == "metadata_missing"]
    numeric_columns = list(perturbed.columns)
    for row_index in missing_rows:
        number_to_mask = max(1, int(round(0.20 * len(numeric_columns))))
        chosen = rng.choice(numeric_columns, size=number_to_mask, replace=False)
        perturbed.loc[row_index, chosen] = np.nan

    shifted_rows = metadata.index[metadata["scenario"] == "distribution_shift"]
    shift_columns = [column for column in COMPLEXITY_COLUMNS if column in perturbed.columns]
    if shift_columns and len(shifted_rows):
        q95 = model_bundle.training_quantiles.loc["q95", shift_columns]
        q50 = model_bundle.training_quantiles.loc["q50", shift_columns]
        increment = (q95 - q50).replace(0, 1.0)
        perturbed.loc[shifted_rows, shift_columns] = (
            perturbed.loc[shifted_rows, shift_columns].fillna(q95) + 1.25 * increment
        )

    metadata["missing_fraction"] = perturbed.isna().mean(axis=1)
    metadata["scenario_injected"] = (metadata["scenario"] != "clean").astype(int)
    return ScenarioBundle(features=perturbed, metadata=metadata)

