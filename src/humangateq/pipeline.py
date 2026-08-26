"""End-to-end HumanGate-Q experiment pipeline."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from .agents import assess_workflows
from .data import (
    DatasetError,
    dataset_summary,
    discover_dataset_csv,
    download_kaggle_dataset,
    load_dataset,
)
from .metrics import (
    bootstrap_policy_intervals,
    evaluate_policies,
    selective_risk_curve,
)
from .modeling import fit_reliability_model
from .policies import (
    ablation_actions,
    all_policy_actions,
    oracle_actions,
    tune_humangateq_thresholds,
)
from .reporting import save_artifacts
from .scenarios import build_scenarios
from .visualization import generate_all_figures


def run_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    data_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    download_if_missing: bool = False,
) -> dict[str, Any]:
    """Run training, scenarios, policies, ablations, statistics, plots, and reporting."""
    started = time.perf_counter()
    root = Path(project_root)
    seed = int(config["project"]["random_seed"])
    effective_config = deepcopy(config)
    raw_directory = root / config["dataset"]["raw_directory"]
    if data_path is None:
        try:
            selected_data_path = discover_dataset_csv(
                raw_directory, config["dataset"]["target_column"]
            )
        except DatasetError:
            if not download_if_missing:
                raise
            selected_data_path = download_kaggle_dataset(
                config["dataset"]["kaggle_slug"], raw_directory
            )
    else:
        selected_data_path = Path(data_path)

    destination = Path(output_directory) if output_directory else root / config["output"]["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    figure_directory = destination / "figures"

    print("[1/8] Loading and validating Kaggle data...")
    data = load_dataset(
        selected_data_path,
        target_column=config["dataset"]["target_column"],
        maximum_rows=int(config["dataset"].get("maximum_rows", 0)),
        random_seed=seed,
    )
    summary = dataset_summary(data)
    try:
        summary["source_path"] = str(selected_data_path.resolve().relative_to(root.resolve()))
    except ValueError:
        summary["source_path"] = selected_data_path.name
    print(f"      {summary['rows']:,} rows; {summary['features']} leakage-safe features")

    print("[2/8] Selecting, training, and calibrating reliability model...")
    model_result = fit_reliability_model(
        data.features,
        data.target,
        config["model"],
        test_fraction=float(config["dataset"]["test_fraction"]),
        calibration_fraction=float(config["dataset"]["calibration_fraction_of_training"]),
        random_seed=seed,
        policy_validation_fraction=float(
            config["dataset"].get("policy_validation_fraction_of_training", 0.0)
        ),
        calibration_bins=int(config["evaluation"].get("calibration_bins", 10)),
    )

    print("[3/8] Selecting policy thresholds on non-test validation workflows...")
    threshold_search = pd.DataFrame()
    tuning_config = effective_config["risk"].get("policy_tuning", {})
    if len(model_result.x_policy_validation) and bool(tuning_config.get("enabled", False)):
        validation_scenario_config = deepcopy(effective_config["scenarios"])
        validation_scenario_config["maximum_test_cases"] = min(
            len(model_result.x_policy_validation),
            int(tuning_config.get("validation_maximum_cases", len(model_result.x_policy_validation))),
        )
        validation_scenarios = build_scenarios(
            model_result.bundle,
            model_result.x_policy_validation,
            model_result.y_policy_validation,
            data.identifiers,
            validation_scenario_config,
            seed + 1000,
        )
        validation_assessment = assess_workflows(
            model_result.bundle,
            validation_scenarios.features,
            validation_scenarios.metadata,
            effective_config["risk"],
        )
        validation_oracle = oracle_actions(validation_assessment)
        tuned_thresholds, threshold_search = tune_humangateq_thresholds(
            validation_assessment,
            validation_oracle,
            effective_config["risk"]["thresholds"],
            tuning_config,
        )
        effective_config["risk"]["thresholds"] = tuned_thresholds

    print("[4/8] Creating controlled high-stakes test workflows...")
    scenarios = build_scenarios(
        model_result.bundle,
        model_result.x_test,
        model_result.y_test,
        data.identifiers,
        effective_config["scenarios"],
        seed,
    )

    print("[5/8] Running risk agents and collaboration policies...")
    assessment = assess_workflows(
        model_result.bundle,
        scenarios.features,
        scenarios.metadata,
        effective_config["risk"],
    )
    oracle = oracle_actions(assessment)
    actions = all_policy_actions(assessment, effective_config["risk"]["thresholds"])
    policy_metrics = evaluate_policies(actions, oracle)
    bootstrap_intervals = bootstrap_policy_intervals(
        actions,
        oracle,
        repetitions=int(config["evaluation"].get("bootstrap_repetitions", 0)),
        random_seed=seed,
    )

    print("[6/8] Running ablation and selective-risk analyses...")
    ablation_action_table, ablation_risk_table = ablation_actions(
        assessment, effective_config["risk"]
    )
    ablation_metrics = evaluate_policies(ablation_action_table, oracle)
    curve = selective_risk_curve(assessment, oracle)

    print("[7/8] Generating paper-ready figures...")
    figures = generate_all_figures(
        figure_directory,
        data.target,
        model_result,
        assessment,
        policy_metrics,
        actions,
        curve,
        ablation_metrics,
        calibration_bins=int(config["evaluation"].get("calibration_bins", 10)),
    )

    runtime = time.perf_counter() - started
    print("[8/8] Saving models, tables, decisions, and reproducibility record...")
    artifacts = save_artifacts(
        destination,
        effective_config,
        summary,
        model_result,
        assessment,
        oracle,
        actions,
        policy_metrics,
        bootstrap_intervals,
        ablation_action_table,
        ablation_risk_table,
        ablation_metrics,
        curve,
        threshold_search,
        runtime,
    )
    return {
        "output_directory": destination,
        "dataset_summary": summary,
        "model_metrics": model_result.metrics,
        "policy_metrics": policy_metrics,
        "ablation_metrics": ablation_metrics,
        "figures": figures,
        "artifacts": artifacts,
        "runtime_seconds": runtime,
    }
