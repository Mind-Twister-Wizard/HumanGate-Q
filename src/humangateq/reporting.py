"""Persist tables, decisions, metadata, and a chapter-ready result summary."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _percentage_points(value: float, signed: bool = False) -> str:
    format_specifier = "+.2f" if signed else ".2f"
    return f"{format(100.0 * float(value), format_specifier)} percentage points"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def write_results_summary(
    path: Path,
    dataset_summary: dict[str, Any],
    model_metrics: dict[str, float],
    policy_metrics: pd.DataFrame,
    ablation_metrics: pd.DataFrame,
    scenario_count: int,
    runtime_seconds: float,
    selected_candidate: str,
) -> None:
    human = policy_metrics.loc["HumanGate-Q"]
    autonomous = policy_metrics.loc["Fully Autonomous"]
    always_human = policy_metrics.loc["Always Human"]
    unsafe_reduction = autonomous["unsafe_execution_rate"] - human["unsafe_execution_rate"]
    review_reduction = always_human["human_review_rate"] - human["human_review_rate"]
    lines = [
        "# HumanGate-Q Verified Results Summary",
        "",
        "> This file is generated from the completed experiment. Use these values—not estimates or unit-test output—in the chapter.",
        "",
        "## Dataset and split",
        "",
        f"- Kaggle rows used: **{dataset_summary['rows']:,}**",
        f"- Leakage-safe structural features: **{dataset_summary['features']:,}**",
        f"- Controlled held-out workflow cases: **{scenario_count:,}**",
        f"- Dataset SHA-256: `{dataset_summary['sha256']}`",
        "",
        "## Reliability model",
        "",
        f"- Training-only cross-validation selected: **{selected_candidate}**",
        f"- Accuracy: **{_percent(model_metrics['accuracy'])}**",
        f"- Balanced accuracy: **{_percent(model_metrics['balanced_accuracy'])}**",
        f"- Macro F1: **{model_metrics['macro_f1']:.4f}**",
        f"- Expected calibration error: **{model_metrics['expected_calibration_error']:.4f}**",
        f"- Multiclass Brier score: **{model_metrics['multiclass_brier']:.4f}**",
        f"- Learned calibration temperature: **{model_metrics['temperature']:.4f}**",
        f"- Original Extra Trees accuracy on the identical test split: **{_percent(model_metrics['baseline_accuracy'])}**",
        f"- Absolute accuracy change: **{_percentage_points(model_metrics['absolute_accuracy_gain'], signed=True)}**",
        "",
        "## Primary policy results",
        "",
        f"- Policy-threshold validation cases (separate from test): **{int(model_metrics.get('policy_validation_rows', 0)):,}**",
        f"- HumanGate-Q exact action accuracy: **{_percent(human['exact_action_accuracy'])}**",
        f"- HumanGate-Q unsafe workflow-execution rate: **{_percent(human['unsafe_execution_rate'])}**",
        f"- HumanGate-Q appropriate escalation recall: **{_percent(human['appropriate_escalation_recall'])}**",
        f"- HumanGate-Q safe automation coverage: **{_percent(human['safe_automation_coverage'])}**",
        f"- HumanGate-Q human-review rate: **{_percent(human['human_review_rate'])}**",
        f"- Absolute unsafe-execution reduction versus full autonomy: **{_percentage_points(unsafe_reduction)}**",
        f"- Absolute human-review reduction versus always-human review: **{_percentage_points(review_reduction)}**",
        "",
        "## Ablation interpretation",
        "",
        "The detailed ablation table is in `tables/ablation_metrics.csv`. Compare every variant with `Full HumanGate-Q`; do not claim that a component helps unless its removal worsens the relevant metric in this run.",
        "",
        "## Runtime",
        "",
        f"- End-to-end experiment runtime: **{runtime_seconds:.2f} seconds**",
        "",
        "## Files for the chapter",
        "",
        "- `tables/policy_metrics.csv` — main comparison",
        "- `tables/model_selection_cv.csv` — training-only candidate selection",
        "- `tables/model_comparison.csv` — original versus upgraded test metrics",
        "- `tables/policy_threshold_search.csv` — validation-only safety-constrained threshold selection",
        "- `tables/bootstrap_confidence_intervals.csv` — 95% intervals",
        "- `tables/ablation_metrics.csv` — component study",
        "- `tables/workflow_decisions.csv` — auditable per-case decisions",
        "- `figures/` — ten high-resolution figures, including the system architecture",
        "",
        "## Required limitation statement",
        "",
        "The experiment uses synthetic circuits and simulated-noise reliability labels from Kaggle. Workflow criticality and failure conditions are controlled perturbations, and the reference human-oversight action is a declared experimental oracle rather than observed human behaviour. Results therefore demonstrate policy behaviour in a reproducible simulation and do not establish real-QPU or real-world safety.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def save_artifacts(
    output_directory: Path,
    config: dict[str, Any],
    dataset_summary: dict[str, Any],
    model_result,
    assessment: pd.DataFrame,
    oracle: pd.Series,
    actions: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    ablation_actions: pd.DataFrame,
    ablation_risks: pd.DataFrame,
    ablation_metrics: pd.DataFrame,
    selective_curve: pd.DataFrame,
    policy_threshold_search: pd.DataFrame,
    runtime_seconds: float,
) -> dict[str, Path]:
    tables = output_directory / "tables"
    models = output_directory / "models"
    tables.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    model_metrics_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in model_result.metrics.items()]
    )
    model_metrics_frame.to_csv(tables / "model_metrics.csv", index=False)
    policy_metrics.to_csv(tables / "policy_metrics.csv")
    bootstrap_intervals.to_csv(tables / "bootstrap_confidence_intervals.csv", index=False)
    ablation_metrics.to_csv(tables / "ablation_metrics.csv")
    selective_curve.to_csv(tables / "risk_coverage_curve.csv", index=False)
    model_result.feature_importance.to_csv(tables / "feature_importance.csv", index=False)
    model_result.model_selection.to_csv(tables / "model_selection_cv.csv", index=False)
    model_result.model_comparison.to_csv(tables / "model_comparison.csv", index=False)
    policy_threshold_search.to_csv(tables / "policy_threshold_search.csv", index=False)

    decisions = assessment.copy()
    decisions["oracle_action"] = oracle
    for policy in actions.columns:
        decisions[f"action__{policy}"] = actions[policy]
    decisions.to_csv(tables / "workflow_decisions.csv", index=False)

    ablation_detail = pd.concat(
        {
            "risk": ablation_risks,
            "action": ablation_actions,
        },
        axis=1,
    )
    ablation_detail.to_csv(tables / "ablation_decisions.csv", index=False)
    joblib.dump(model_result.bundle, models / "humangateq_reliability_model.joblib")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime_seconds,
        "dataset": dataset_summary,
        "configuration": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": _package_version("lightgbm"),
        },
        "model_classes": model_result.bundle.classes,
        "model_temperature": model_result.bundle.temperature,
        "model_backend": model_result.bundle.model_backend,
        "selected_candidate": model_result.bundle.selected_candidate,
    }
    write_json(output_directory / "run_metadata.json", metadata)
    write_results_summary(
        output_directory / "PAPER_RESULTS_SUMMARY.md",
        dataset_summary,
        model_result.metrics,
        policy_metrics,
        ablation_metrics,
        len(assessment),
        runtime_seconds,
        model_result.bundle.selected_candidate,
    )
    return {
        "summary": output_directory / "PAPER_RESULTS_SUMMARY.md",
        "policy_metrics": tables / "policy_metrics.csv",
        "model_comparison": tables / "model_comparison.csv",
        "policy_threshold_search": tables / "policy_threshold_search.csv",
        "decisions": tables / "workflow_decisions.csv",
        "model": models / "humangateq_reliability_model.joblib",
        "metadata": output_directory / "run_metadata.json",
    }
