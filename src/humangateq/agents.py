"""Cooperating HumanGate-Q risk-assessment agents."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .modeling import (
    ReliabilityModelBundle,
    calibrated_probabilities,
    out_of_distribution_score,
    resource_complexity_score,
)

RISK_COMPONENTS = (
    "reliability_risk",
    "predictive_uncertainty",
    "distribution_shift_risk",
    "verification_risk",
    "resource_complexity_risk",
    "workflow_criticality",
)


def _probability_column(probabilities: pd.DataFrame, label: str) -> pd.Series:
    name = f"probability_{label.lower()}"
    if name in probabilities.columns:
        return probabilities[name]
    return pd.Series(0.0, index=probabilities.index, name=name)


def normalized_entropy(probabilities: pd.DataFrame) -> pd.Series:
    values = np.clip(probabilities.to_numpy(), 1e-12, 1.0)
    entropy = -np.sum(values * np.log(values), axis=1)
    denominator = np.log(values.shape[1]) if values.shape[1] > 1 else 1.0
    return pd.Series(entropy / denominator, index=probabilities.index, name="predictive_uncertainty")


def verification_risk(metadata: pd.DataFrame) -> pd.Series:
    ambiguous = metadata["ambiguous_goal"].astype(float) * 0.60
    tool_failure = metadata["tool_failure"].astype(float)
    missing = (metadata["missing_fraction"].astype(float) * 3.0).clip(upper=1.0)
    shifted = (metadata["scenario"] == "distribution_shift").astype(float) * 0.45
    return pd.concat([ambiguous, tool_failure, missing, shifted], axis=1).max(axis=1).rename(
        "verification_risk"
    )


def fuse_risk(assessment: pd.DataFrame, risk_config: dict[str, Any]) -> pd.Series:
    weights = risk_config["weights"]
    interactions = risk_config.get("interactions", {})
    risk = (
        float(weights["reliability"]) * assessment["reliability_risk"]
        + float(weights["predictive_uncertainty"]) * assessment["predictive_uncertainty"]
        + float(weights["distribution_shift"]) * assessment["distribution_shift_risk"]
        + float(weights["verification"]) * assessment["verification_risk"]
        + float(weights["resource_complexity"]) * assessment["resource_complexity_risk"]
        + float(weights["workflow_criticality"]) * assessment["workflow_criticality"]
    )
    risk += float(interactions.get("reliability_x_criticality", 0.0)) * (
        assessment["reliability_risk"] * assessment["workflow_criticality"]
    )
    risk += float(interactions.get("verification_x_criticality", 0.0)) * (
        assessment["verification_risk"] * assessment["workflow_criticality"]
    )
    if "tool_failure" in assessment:
        risk = risk.where(assessment["tool_failure"] == 0, np.maximum(risk, 0.95))
    return risk.clip(0.0, 1.0).rename("risk_score")


def assess_workflows(
    model_bundle: ReliabilityModelBundle,
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    risk_config: dict[str, Any],
) -> pd.DataFrame:
    """Run all specialist agents and return one auditable assessment table."""
    probabilities = calibrated_probabilities(model_bundle, features)
    assessment = metadata.copy()
    assessment = assessment.join(probabilities)
    assessment["predicted_reliability"] = probabilities.idxmax(axis=1).str.replace(
        "probability_", "", regex=False
    ).str.upper()
    assessment["model_confidence"] = probabilities.max(axis=1)
    assessment["reliability_risk"] = _probability_column(probabilities, "LOW") + 0.45 * _probability_column(
        probabilities, "MEDIUM"
    )
    assessment["predictive_uncertainty"] = normalized_entropy(probabilities)
    assessment["distribution_shift_risk"] = out_of_distribution_score(model_bundle, features)
    assessment["verification_risk"] = verification_risk(metadata)
    assessment["resource_complexity_risk"] = resource_complexity_score(model_bundle, features)
    assessment["risk_score"] = fuse_risk(assessment, risk_config)

    weighted = pd.DataFrame(index=assessment.index)
    weights = risk_config["weights"]
    for component, config_name in (
        ("reliability_risk", "reliability"),
        ("predictive_uncertainty", "predictive_uncertainty"),
        ("distribution_shift_risk", "distribution_shift"),
        ("verification_risk", "verification"),
        ("resource_complexity_risk", "resource_complexity"),
        ("workflow_criticality", "workflow_criticality"),
    ):
        weighted[component] = assessment[component] * float(weights[config_name])
    assessment["dominant_risk_driver"] = weighted.idxmax(axis=1)
    return assessment

