"""Leakage-safe reliability modeling, probability calibration, and OOD scoring."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class ReliabilityModelBundle:
    pipeline: Pipeline
    classes: list[str]
    temperature: float
    feature_columns: list[str]
    training_quantiles: pd.DataFrame
    random_seed: int
    model_backend: str = "extra_trees"
    selected_candidate: str = "Extra Trees"


@dataclass
class ModelFitResult:
    bundle: ReliabilityModelBundle
    x_train: pd.DataFrame
    y_train: pd.Series
    x_calibration: pd.DataFrame
    y_calibration: pd.Series
    x_policy_validation: pd.DataFrame
    y_policy_validation: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    test_probabilities: pd.DataFrame
    test_predictions: pd.Series
    metrics: dict[str, float]
    confusion: np.ndarray
    feature_importance: pd.DataFrame
    model_selection: pd.DataFrame
    model_comparison: pd.DataFrame


def _softmax(log_values: np.ndarray) -> np.ndarray:
    shifted = log_values - np.max(log_values, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return _softmax(np.log(clipped) / max(float(temperature), 1e-6))


def fit_temperature(probabilities: np.ndarray, labels: pd.Series, classes: list[str]) -> float:
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = np.array([class_to_index[str(label)] for label in labels], dtype=int)

    def objective(temperature: float) -> float:
        calibrated = apply_temperature(probabilities, temperature)
        return float(log_loss(encoded, calibrated, labels=np.arange(len(classes))))

    try:
        result = minimize_scalar(objective, bounds=(0.25, 5.0), method="bounded")
        return float(result.x) if result.success else 1.0
    except (ValueError, FloatingPointError):
        return 1.0


def expected_calibration_error(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: list[str],
    bins: int = 10,
) -> float:
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    true_indices = np.array([classes.index(str(label)) for label in y_true], dtype=int)
    correctness = (predictions == true_indices).astype(float)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(float(correctness[mask].mean() - confidence[mask].mean()))
    return float(ece)


def multiclass_brier_score(y_true: pd.Series, probabilities: np.ndarray, classes: list[str]) -> float:
    encoded = np.array([classes.index(str(label)) for label in y_true], dtype=int)
    one_hot = np.eye(len(classes))[encoded]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def _model_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: list[str],
    calibration_bins: int,
) -> tuple[dict[str, float], pd.Series, np.ndarray]:
    predicted_indices = np.argmax(probabilities, axis=1)
    predictions = pd.Series([classes[index] for index in predicted_indices], index=y_true.index)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "multiclass_brier": multiclass_brier_score(y_true, probabilities, classes),
        "expected_calibration_error": expected_calibration_error(
            y_true, probabilities, classes, bins=calibration_bins
        ),
    }
    matrix = confusion_matrix(y_true, predictions, labels=classes)
    return metrics, predictions, matrix


def _imputed_pipeline(classifier: Any) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True),
            ),
            ("classifier", classifier),
        ]
    )


def _extra_trees_pipeline(model_config: dict[str, Any], random_seed: int) -> Pipeline:
    baseline = model_config.get("baseline_extra_trees", model_config)
    classifier = ExtraTreesClassifier(
        n_estimators=int(baseline.get("n_estimators", model_config.get("n_estimators", 160))),
        max_depth=baseline.get("max_depth", model_config.get("max_depth")),
        min_samples_leaf=int(
            baseline.get("min_samples_leaf", model_config.get("min_samples_leaf", 2))
        ),
        class_weight=baseline.get("class_weight", model_config.get("class_weight", "balanced")),
        n_jobs=int(baseline.get("n_jobs", model_config.get("n_jobs", -1))),
        random_state=int(random_seed),
    )
    return _imputed_pipeline(classifier)


def _lightgbm_pipeline(
    parameters: dict[str, Any],
    random_seed: int,
    n_jobs: int,
) -> Pipeline:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise RuntimeError(
            "The accuracy-upgrade backend requires LightGBM. Run START_HERE.bat "
            "or install the package requirements."
        ) from exc

    classifier = LGBMClassifier(
        objective="multiclass",
        n_jobs=int(n_jobs),
        random_state=int(random_seed),
        verbosity=-1,
        importance_type="gain",
        **parameters,
    )
    return _imputed_pipeline(classifier)


def _select_lightgbm_candidate(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    model_config: dict[str, Any],
    random_seed: int,
) -> tuple[Pipeline, str, pd.DataFrame]:
    """Select a LightGBM configuration using training-only stratified CV."""
    candidates = model_config.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("model.candidates must contain at least one LightGBM configuration.")

    folds = max(2, int(model_config.get("validation_folds", 3)))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(random_seed))
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    rows: list[dict[str, Any]] = []
    n_jobs = int(model_config.get("n_jobs", -1))

    for position, configured in enumerate(candidates):
        if not isinstance(configured, dict):
            raise ValueError("Every model candidate must be a mapping.")
        name = str(configured.get("name", f"candidate_{position + 1}"))
        parameters = {key: value for key, value in configured.items() if key != "name"}
        pipeline = _lightgbm_pipeline(parameters, random_seed, n_jobs)
        started = time.perf_counter()
        result = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        row: dict[str, Any] = {
            "candidate": name,
            "cv_folds": folds,
            "cv_accuracy_mean": float(np.mean(result["test_accuracy"])),
            "cv_accuracy_std": float(np.std(result["test_accuracy"])),
            "cv_balanced_accuracy_mean": float(np.mean(result["test_balanced_accuracy"])),
            "cv_macro_f1_mean": float(np.mean(result["test_macro_f1"])),
            "cv_fit_seconds_mean": float(np.mean(result["fit_time"])),
            "search_seconds": float(time.perf_counter() - started),
        }
        for key, value in parameters.items():
            row[f"parameter__{key}"] = value
        rows.append(row)

    selection_metric = str(model_config.get("selection_metric", "accuracy")).lower()
    metric_column = {
        "accuracy": "cv_accuracy_mean",
        "balanced_accuracy": "cv_balanced_accuracy_mean",
        "macro_f1": "cv_macro_f1_mean",
    }.get(selection_metric)
    if metric_column is None:
        raise ValueError(
            "model.selection_metric must be accuracy, balanced_accuracy, or macro_f1."
        )

    table = (
        pd.DataFrame(rows)
        .sort_values(
            [metric_column, "cv_balanced_accuracy_mean", "cv_macro_f1_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    selected_name = str(table.loc[0, "candidate"])
    table["selected"] = table["candidate"].eq(selected_name)
    selected_configuration = next(
        configured for configured in candidates if str(configured.get("name")) == selected_name
    )
    selected_parameters = {
        key: value for key, value in selected_configuration.items() if key != "name"
    }
    selected_pipeline = _lightgbm_pipeline(selected_parameters, random_seed, n_jobs)
    return selected_pipeline, selected_name, table


def _calibrated_test_evaluation(
    pipeline: Pipeline,
    x_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    calibration_bins: int,
) -> tuple[list[str], float, np.ndarray, dict[str, float], pd.Series, np.ndarray]:
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    calibration_raw = pipeline.predict_proba(x_calibration)
    temperature = fit_temperature(calibration_raw, y_calibration, classes)
    probabilities = apply_temperature(pipeline.predict_proba(x_test), temperature)
    metrics, predictions, matrix = _model_metrics(
        y_test, probabilities, classes, calibration_bins
    )
    metrics["temperature"] = float(temperature)
    return classes, float(temperature), probabilities, metrics, predictions, matrix


def fit_reliability_model(
    features: pd.DataFrame,
    target: pd.Series,
    model_config: dict[str, Any],
    test_fraction: float,
    calibration_fraction: float,
    random_seed: int,
    policy_validation_fraction: float = 0.0,
    calibration_bins: int = 10,
) -> ModelFitResult:
    """Select, fit, and calibrate a leakage-safe reliability classifier."""
    x_train_cal, x_test, y_train_cal, y_test = train_test_split(
        features,
        target,
        test_size=float(test_fraction),
        random_state=int(random_seed),
        stratify=target,
    )
    reserved_fraction = float(calibration_fraction) + float(policy_validation_fraction)
    if not 0.0 < reserved_fraction < 1.0:
        raise ValueError(
            "Calibration and policy-validation fractions must sum to a value between 0 and 1."
        )
    x_train, x_reserved, y_train, y_reserved = train_test_split(
        x_train_cal,
        y_train_cal,
        test_size=reserved_fraction,
        random_state=int(random_seed) + 1,
        stratify=y_train_cal,
    )
    if float(policy_validation_fraction) > 0.0:
        policy_share = float(policy_validation_fraction) / reserved_fraction
        x_cal, x_policy, y_cal, y_policy = train_test_split(
            x_reserved,
            y_reserved,
            test_size=policy_share,
            random_state=int(random_seed) + 2,
            stratify=y_reserved,
        )
    else:
        x_cal, y_cal = x_reserved, y_reserved
        x_policy = x_reserved.iloc[0:0].copy()
        y_policy = y_reserved.iloc[0:0].copy()

    backend = str(model_config.get("backend", "extra_trees")).lower()
    if backend == "lightgbm_search":
        pipeline, selected_candidate, model_selection = _select_lightgbm_candidate(
            x_train, y_train, model_config, random_seed
        )
    elif backend == "extra_trees":
        pipeline = _extra_trees_pipeline(model_config, random_seed)
        selected_candidate = "Extra Trees"
        model_selection = pd.DataFrame(
            [
                {
                    "rank": 1,
                    "candidate": selected_candidate,
                    "selected": True,
                    "cv_folds": 0,
                }
            ]
        )
    else:
        raise ValueError("model.backend must be 'lightgbm_search' or 'extra_trees'.")

    pipeline.fit(x_train, y_train)
    (
        classes,
        temperature,
        test_probabilities_array,
        metrics,
        test_predictions,
        matrix,
    ) = _calibrated_test_evaluation(
        pipeline, x_cal, y_cal, x_test, y_test, calibration_bins
    )
    test_probabilities = pd.DataFrame(
        test_probabilities_array,
        index=x_test.index,
        columns=[f"probability_{label.lower()}" for label in classes],
    )

    if backend == "lightgbm_search":
        baseline_pipeline = _extra_trees_pipeline(model_config, random_seed)
        baseline_pipeline.fit(x_train, y_train)
        (
            _,
            baseline_temperature,
            _,
            baseline_metrics,
            _,
            _,
        ) = _calibrated_test_evaluation(
            baseline_pipeline, x_cal, y_cal, x_test, y_test, calibration_bins
        )
    else:
        baseline_temperature = temperature
        baseline_metrics = metrics.copy()

    metrics["baseline_accuracy"] = float(baseline_metrics["accuracy"])
    metrics["absolute_accuracy_gain"] = float(
        metrics["accuracy"] - baseline_metrics["accuracy"]
    )
    metrics["baseline_balanced_accuracy"] = float(baseline_metrics["balanced_accuracy"])
    metrics["absolute_balanced_accuracy_gain"] = float(
        metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
    )
    selected_row = model_selection.loc[model_selection["selected"]].iloc[0]
    if "cv_accuracy_mean" in selected_row:
        metrics["selected_cv_accuracy_mean"] = float(selected_row["cv_accuracy_mean"])
    metrics["training_rows"] = float(len(x_train))
    metrics["calibration_rows"] = float(len(x_cal))
    metrics["policy_validation_rows"] = float(len(x_policy))
    metrics["test_rows"] = float(len(x_test))

    quantiles = x_train.quantile([0.05, 0.50, 0.95], numeric_only=True)
    quantiles.index = ["q05", "q50", "q95"]
    importances = np.asarray(
        pipeline.named_steps["classifier"].feature_importances_, dtype=float
    )
    if float(importances.sum()) > 0:
        importances = importances / float(importances.sum())
    feature_importance = (
        pd.DataFrame({"feature": list(features.columns), "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    model_comparison = pd.DataFrame(
        [
            {
                "model": "Original Extra Trees",
                "role": "baseline",
                "accuracy": baseline_metrics["accuracy"],
                "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                "macro_f1": baseline_metrics["macro_f1"],
                "log_loss": baseline_metrics["log_loss"],
                "multiclass_brier": baseline_metrics["multiclass_brier"],
                "expected_calibration_error": baseline_metrics[
                    "expected_calibration_error"
                ],
                "temperature": baseline_temperature,
            },
            {
                "model": selected_candidate,
                "role": "selected",
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "multiclass_brier": metrics["multiclass_brier"],
                "expected_calibration_error": metrics["expected_calibration_error"],
                "temperature": temperature,
            },
        ]
    )
    bundle = ReliabilityModelBundle(
        pipeline=pipeline,
        classes=classes,
        temperature=float(temperature),
        feature_columns=list(features.columns),
        training_quantiles=quantiles,
        random_seed=int(random_seed),
        model_backend=backend,
        selected_candidate=selected_candidate,
    )
    return ModelFitResult(
        bundle=bundle,
        x_train=x_train,
        y_train=y_train,
        x_calibration=x_cal,
        y_calibration=y_cal,
        x_policy_validation=x_policy,
        y_policy_validation=y_policy,
        x_test=x_test,
        y_test=y_test,
        test_probabilities=test_probabilities,
        test_predictions=test_predictions,
        metrics=metrics,
        confusion=matrix,
        feature_importance=feature_importance,
        model_selection=model_selection,
        model_comparison=model_comparison,
    )


def calibrated_probabilities(bundle: ReliabilityModelBundle, features: pd.DataFrame) -> pd.DataFrame:
    ordered = features.reindex(columns=bundle.feature_columns)
    raw = bundle.pipeline.predict_proba(ordered)
    calibrated = apply_temperature(raw, bundle.temperature)
    return pd.DataFrame(
        calibrated,
        index=features.index,
        columns=[f"probability_{label.lower()}" for label in bundle.classes],
    )


def out_of_distribution_score(bundle: ReliabilityModelBundle, features: pd.DataFrame) -> pd.Series:
    """Robustly score feature excursions outside training 5%-95% intervals."""
    ordered = features.reindex(columns=bundle.feature_columns)
    q05 = bundle.training_quantiles.loc["q05"].reindex(bundle.feature_columns)
    q50 = bundle.training_quantiles.loc["q50"].reindex(bundle.feature_columns)
    q95 = bundle.training_quantiles.loc["q95"].reindex(bundle.feature_columns)
    filled = ordered.fillna(q50)
    scale = (q95 - q05).replace(0, np.nan).fillna(1.0)
    lower_excursion = ((q05 - filled) / scale).clip(lower=0)
    upper_excursion = ((filled - q95) / scale).clip(lower=0)
    excursion = lower_excursion + upper_excursion
    top_count = max(1, min(5, excursion.shape[1]))
    top_values = np.sort(excursion.to_numpy(), axis=1)[:, -top_count:]
    raw_distance = np.mean(top_values, axis=1)
    score = 1.0 - np.exp(-raw_distance)
    return pd.Series(np.clip(score, 0.0, 1.0), index=features.index, name="distribution_shift_risk")


COMPLEXITY_COLUMNS = (
    "depth",
    "total_operations",
    "two_qubit_gates",
    "three_qubit_gates",
    "entangling_gates",
    "cx_depth",
    "number_of_qubits",
)


def resource_complexity_score(bundle: ReliabilityModelBundle, features: pd.DataFrame) -> pd.Series:
    available = [column for column in COMPLEXITY_COLUMNS if column in features.columns]
    if not available:
        return pd.Series(0.0, index=features.index, name="resource_complexity_risk")
    q50 = bundle.training_quantiles.loc["q50", available]
    q95 = bundle.training_quantiles.loc["q95", available]
    scale = (q95 - q50).replace(0, 1.0)
    normalized = (features[available].fillna(q50) - q50) / scale
    return normalized.clip(lower=0.0, upper=1.0).mean(axis=1).rename("resource_complexity_risk")
