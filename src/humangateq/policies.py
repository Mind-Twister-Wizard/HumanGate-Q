"""HumanGate-Q action policy, transparent oracle, baselines, and ablations."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from .agents import fuse_risk

EXECUTE = "EXECUTE"
SELF_REPAIR = "SELF_REPAIR"
ASK_HUMAN = "ASK_HUMAN"
ABSTAIN = "ABSTAIN"
ACTIONS = (EXECUTE, SELF_REPAIR, ASK_HUMAN, ABSTAIN)


def oracle_actions(assessment: pd.DataFrame) -> pd.Series:
    """Construct the declared controlled reference action for policy evaluation."""
    actions: list[str] = []
    for row in assessment.itertuples():
        label = str(row.true_reliability).upper()
        criticality = float(row.workflow_criticality)
        scenario = str(row.scenario)
        if int(row.tool_failure) == 1 or label == "LOW":
            action = ABSTAIN
        elif scenario in {"ambiguous_goal", "metadata_missing", "distribution_shift"}:
            action = ASK_HUMAN if criticality >= 0.55 or label == "MEDIUM" else SELF_REPAIR
        elif label == "MEDIUM":
            action = ASK_HUMAN if criticality >= 0.65 else SELF_REPAIR
        elif label == "HIGH" and criticality >= 0.95:
            action = ASK_HUMAN
        else:
            action = EXECUTE
        actions.append(action)
    return pd.Series(actions, index=assessment.index, name="oracle_action")


def humangateq_actions(assessment: pd.DataFrame, thresholds: dict[str, Any]) -> pd.Series:
    repair = float(thresholds["repair"])
    review = float(thresholds["human_review"])
    abstain = float(thresholds["abstain"])
    high_review = float(thresholds["high_stakes_review"])
    high_criticality = float(thresholds["high_stakes_criticality"])
    low_probability_abstain = float(thresholds["direct_low_probability_abstain"])
    p_low = assessment.get("probability_low", pd.Series(0.0, index=assessment.index))

    risk = assessment["risk_score"].to_numpy(dtype=float)
    criticality = assessment["workflow_criticality"].to_numpy(dtype=float)
    tool_failure = assessment["tool_failure"].to_numpy(dtype=int)
    low_probability = p_low.to_numpy(dtype=float)

    decisions = np.full(len(assessment), EXECUTE, dtype=object)
    decisions[risk >= repair] = SELF_REPAIR
    review_mask = (risk >= review) | (
        (criticality >= high_criticality) & (risk >= high_review)
    )
    decisions[review_mask] = ASK_HUMAN
    abstain_mask = (
        (tool_failure == 1)
        | (risk >= abstain)
        | (low_probability >= low_probability_abstain)
    )
    decisions[abstain_mask] = ABSTAIN
    return pd.Series(decisions, index=assessment.index, name="HumanGate-Q")


def tune_humangateq_thresholds(
    assessment: pd.DataFrame,
    oracle: pd.Series,
    base_thresholds: dict[str, Any],
    tuning_config: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    """Select policy thresholds on a non-test validation set under safety constraints."""
    from .metrics import evaluate_policy

    if not bool(tuning_config.get("enabled", False)):
        actions = humangateq_actions(assessment, base_thresholds)
        metrics = evaluate_policy(actions, oracle)
        row = {"selected": True, "feasible": True, **base_thresholds, **metrics}
        return {key: float(value) for key, value in base_thresholds.items()}, pd.DataFrame([row])

    grid = tuning_config.get("grid", {})

    def values(name: str) -> list[float]:
        configured = grid.get(name, [base_thresholds[name]])
        if not isinstance(configured, list):
            configured = [configured]
        return [float(value) for value in configured]

    names = (
        "repair",
        "human_review",
        "abstain",
        "high_stakes_review",
        "direct_low_probability_abstain",
    )
    fixed_criticality = float(base_thresholds["high_stakes_criticality"])
    target_unsafe = float(tuning_config.get("target_unsafe_execution_rate", 0.13))
    minimum_recall = float(tuning_config.get("minimum_escalation_recall", 0.86))
    minimum_safe_coverage = float(
        tuning_config.get("minimum_safe_automation_coverage", 0.88)
    )
    maximum_review = float(tuning_config.get("maximum_human_review_rate", 0.30))
    records: list[dict[str, Any]] = []

    for combination in product(*(values(name) for name in names)):
        candidate = {
            name: float(value) for name, value in zip(names, combination)
        }
        candidate["high_stakes_criticality"] = fixed_criticality
        if not (
            0.0
            <= candidate["repair"]
            <= candidate["human_review"]
            <= candidate["abstain"]
            <= 1.0
        ):
            continue
        actions = humangateq_actions(assessment, candidate)
        metrics = evaluate_policy(actions, oracle)
        feasible = (
            metrics["unsafe_execution_rate"] <= target_unsafe
            and metrics["appropriate_escalation_recall"] >= minimum_recall
            and metrics["safe_automation_coverage"] >= minimum_safe_coverage
            and metrics["human_review_rate"] <= maximum_review
        )
        violation = (
            4.0 * max(0.0, metrics["unsafe_execution_rate"] - target_unsafe)
            + 3.0 * max(0.0, minimum_recall - metrics["appropriate_escalation_recall"])
            + 2.0 * max(0.0, minimum_safe_coverage - metrics["safe_automation_coverage"])
            + max(0.0, metrics["human_review_rate"] - maximum_review)
        )
        records.append(
            {
                **candidate,
                **metrics,
                "feasible": bool(feasible),
                "constraint_violation": float(violation),
            }
        )

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("The policy-tuning grid did not contain a valid ordered threshold set.")
    table = table.sort_values(
        [
            "feasible",
            "constraint_violation",
            "exact_action_accuracy",
            "unsafe_execution_rate",
            "human_review_rate",
        ],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table["selected"] = False
    table.loc[0, "selected"] = True
    selected = {key: float(table.loc[0, key]) for key in base_thresholds}
    return selected, table


def baseline_actions(assessment: pd.DataFrame) -> dict[str, pd.Series]:
    index = assessment.index
    confidence = assessment["model_confidence"]
    predicted = assessment["predicted_reliability"]
    verification = assessment["verification_risk"]
    criticality = assessment["workflow_criticality"]

    confidence_only = pd.Series(EXECUTE, index=index, name="Confidence Only")
    confidence_only.loc[confidence < 0.70] = ASK_HUMAN
    confidence_only.loc[(predicted == "LOW") & (confidence >= 0.70)] = ABSTAIN

    criticality_only = pd.Series(EXECUTE, index=index, name="Criticality Only")
    criticality_only.loc[criticality >= 0.70] = ASK_HUMAN

    verifier_only = pd.Series(EXECUTE, index=index, name="Verifier Only")
    verifier_only.loc[(verification >= 0.45) & (assessment["tool_failure"] == 0)] = ASK_HUMAN
    verifier_only.loc[assessment["tool_failure"] == 1] = ABSTAIN

    return {
        "Fully Autonomous": pd.Series(EXECUTE, index=index, name="Fully Autonomous"),
        "Always Human": pd.Series(ASK_HUMAN, index=index, name="Always Human"),
        "Confidence Only": confidence_only,
        "Criticality Only": criticality_only,
        "Verifier Only": verifier_only,
    }


def all_policy_actions(
    assessment: pd.DataFrame,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    actions = baseline_actions(assessment)
    actions["HumanGate-Q"] = humangateq_actions(assessment, thresholds)
    return pd.DataFrame(actions, index=assessment.index)


def ablation_actions(
    assessment: pd.DataFrame,
    risk_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute risk and decisions for the chapter's declared ablation variants.

    The two distribution-shift pathways are separated deliberately: the
    structural out-of-distribution score is a weighted risk input, while the
    explicit scenario flag contributes to verifier risk. This distinction is
    preserved in the published ablation table.
    """
    variants = (
        ("Full HumanGate-Q", (), False, False),
        ("Without reliability", ("reliability",), False, False),
        (
            "Without predictive uncertainty",
            ("predictive_uncertainty",),
            False,
            False,
        ),
        (
            "Without structural shift",
            ("distribution_shift",),
            False,
            False,
        ),
        ("Without explicit shift flag", (), False, True),
        (
            "Without both shift pathways",
            ("distribution_shift",),
            False,
            True,
        ),
        ("Without verification", ("verification",), False, False),
        ("Without resource risk", ("resource_complexity",), False, False),
        ("Without criticality", ("workflow_criticality",), False, False),
        ("Without interactions", (), True, False),
    )
    actions = pd.DataFrame(index=assessment.index)
    risks = pd.DataFrame(index=assessment.index)
    for name, zero_weights, zero_interactions, remove_shift_flag in variants:
        config = deepcopy(risk_config)
        for component in zero_weights:
            config["weights"][component] = 0.0
        if zero_interactions:
            config["interactions"] = {
                key: 0.0 for key in config.get("interactions", {})
            }
        variant = assessment.copy()
        if remove_shift_flag:
            required = {
                "ambiguous_goal",
                "tool_failure",
                "missing_fraction",
            }
            missing = required.difference(variant.columns)
            if missing:
                raise ValueError(
                    "The explicit-shift ablation requires assessment columns: "
                    + ", ".join(sorted(missing))
                )
            ambiguous = variant["ambiguous_goal"].astype(float) * 0.60
            tool_failure = variant["tool_failure"].astype(float)
            metadata_missing = (
                variant["missing_fraction"].astype(float) * 3.0
            ).clip(upper=1.0)
            variant["verification_risk"] = pd.concat(
                [ambiguous, tool_failure, metadata_missing], axis=1
            ).max(axis=1)
        variant["risk_score"] = fuse_risk(variant, config)
        risks[name] = variant["risk_score"]
        actions[name] = humangateq_actions(variant, risk_config["thresholds"])
    return actions, risks
