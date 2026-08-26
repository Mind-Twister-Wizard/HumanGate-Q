"""Policy metrics, confidence intervals, and selective risk curves."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .policies import ABSTAIN, ASK_HUMAN, EXECUTE, SELF_REPAIR

AUTOMATED = {EXECUTE, SELF_REPAIR}
ESCALATED = {ASK_HUMAN, ABSTAIN}


def evaluate_policy(predicted: pd.Series, oracle: pd.Series) -> dict[str, float]:
    predicted = predicted.astype(str)
    oracle = oracle.astype(str)
    automated = predicted.isin(AUTOMATED)
    escalation = predicted.isin(ESCALATED)
    oracle_safe_auto = oracle.isin(AUTOMATED)
    oracle_escalate = oracle.isin(ESCALATED)
    execute = predicted.eq(EXECUTE)
    severe_oracle = oracle.eq(ABSTAIN)

    unsafe_auto = automated & oracle_escalate
    unsafe_execute = execute & oracle_escalate
    severe_unsafe = execute & severe_oracle
    appropriate_escalation = escalation & oracle_escalate
    safe_auto = automated & oracle_safe_auto
    over_escalation = escalation & oracle_safe_auto

    def conditional_rate(numerator: pd.Series, condition: pd.Series) -> float:
        denominator = int(condition.sum())
        return float((numerator & condition).sum() / denominator) if denominator else 0.0

    return {
        "exact_action_accuracy": float((predicted == oracle).mean()),
        "unsafe_automation_rate": conditional_rate(unsafe_auto, automated),
        "unsafe_execution_rate": conditional_rate(unsafe_execute, execute),
        "severe_unsafe_execution_rate": conditional_rate(severe_unsafe, execute),
        "appropriate_escalation_recall": conditional_rate(appropriate_escalation, oracle_escalate),
        "safe_automation_coverage": conditional_rate(safe_auto, oracle_safe_auto),
        "over_escalation_rate": conditional_rate(over_escalation, oracle_safe_auto),
        "automation_rate": float(automated.mean()),
        "human_review_rate": float(predicted.eq(ASK_HUMAN).mean()),
        "abstention_rate": float(predicted.eq(ABSTAIN).mean()),
        "self_repair_rate": float(predicted.eq(SELF_REPAIR).mean()),
        "execute_rate": float(predicted.eq(EXECUTE).mean()),
    }


def evaluate_policies(actions: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    rows = []
    for policy in actions.columns:
        row = {"policy": policy}
        row.update(evaluate_policy(actions[policy], oracle))
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


def bootstrap_policy_intervals(
    actions: pd.DataFrame,
    oracle: pd.Series,
    repetitions: int,
    random_seed: int,
    metrics: Iterable[str] = (
        "unsafe_execution_rate",
        "human_review_rate",
        "appropriate_escalation_recall",
        "safe_automation_coverage",
    ),
) -> pd.DataFrame:
    """Paired nonparametric bootstrap intervals for headline policy metrics."""
    if repetitions <= 0:
        return pd.DataFrame(
            columns=["policy", "metric", "estimate", "ci_lower", "ci_upper"]
        )
    rng = np.random.default_rng(random_seed + 91)
    positions = np.arange(len(oracle))
    records: list[dict[str, float | str]] = []
    point = evaluate_policies(actions, oracle)
    distributions: dict[tuple[str, str], list[float]] = {
        (policy, metric): [] for policy in actions.columns for metric in metrics
    }
    for _ in range(repetitions):
        sampled = rng.choice(positions, size=len(positions), replace=True)
        sampled_oracle = oracle.iloc[sampled].reset_index(drop=True)
        sampled_actions = actions.iloc[sampled].reset_index(drop=True)
        result = evaluate_policies(sampled_actions, sampled_oracle)
        for policy in actions.columns:
            for metric in metrics:
                distributions[(policy, metric)].append(float(result.loc[policy, metric]))
    for (policy, metric), values in distributions.items():
        records.append(
            {
                "policy": policy,
                "metric": metric,
                "estimate": float(point.loc[policy, metric]),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            }
        )
    return pd.DataFrame(records)


def selective_risk_curve(assessment: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    """Measure unsafe autonomous coverage while sweeping a human-review threshold."""
    thresholds = np.linspace(0.05, 0.95, 37)
    safe_actions = oracle.isin(AUTOMATED)
    rows = []
    for threshold in thresholds:
        automate = assessment["risk_score"] < threshold
        unsafe = automate & ~safe_actions
        rows.append(
            {
                "risk_threshold": float(threshold),
                "automation_coverage": float(automate.mean()),
                "selective_risk": float(unsafe.sum() / automate.sum()) if automate.any() else 0.0,
            }
        )
    return pd.DataFrame(rows)
