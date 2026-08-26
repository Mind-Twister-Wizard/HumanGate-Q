"""Paper-ready visualizations for HumanGate-Q."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "humangateq_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .policies import ACTIONS

PALETTE = {
    "navy": "#16324F",
    "blue": "#2E86AB",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "orange": "#F4A261",
    "red": "#E76F51",
    "gray": "#6C757D",
}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263238",
            "text.color": "#263238",
        }
    )


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_architecture(path: Path) -> None:
    """Render a precise, publication-ready HumanGate-Q architecture diagram."""
    _style()
    _, axis = plt.subplots(figsize=(12.0, 7.2))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 8.5)
    axis.axis("off")

    def box(x: float, y: float, width: float, height: float, text: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.5,
            edgecolor=PALETTE["navy"],
            facecolor=color,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=10.5,
            weight="bold",
        )

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.3,
                color=PALETTE["gray"],
            )
        )

    box(4.3, 7.0, 3.4, 0.65, "Kaggle quantum workflow record", "#EAF2F8")
    agent_specs = [
        (0.35, 5.30, "Reliability\nAgent", "#D9F0EA"),
        (2.70, 5.30, "Uncertainty\nAgent", "#DDEBF7"),
        (5.05, 5.30, "Verification\nAgent", "#FCE8DE"),
        (7.40, 5.30, "Criticality\nAgent", "#FFF3CD"),
        (9.75, 5.30, "Resource\nAgent", "#E8E1F4"),
    ]
    for x, y, label, color in agent_specs:
        box(x, y, 1.9, 0.9, label, color)
        arrow((6.0, 7.0), (x + 0.95, y + 0.92))
        arrow((x + 0.95, y), (6.0, 4.35))

    box(4.45, 3.55, 3.1, 0.8, "Risk fusion + interaction terms", "#D6EAF8")
    arrow((6.0, 3.55), (6.0, 2.85))
    box(4.55, 2.05, 2.9, 0.8, "Human-Gate Orchestrator", "#CFE8E0")

    action_specs = [
        (0.45, 0.35, "EXECUTE", "#D9F0EA"),
        (3.05, 0.35, "SELF-REPAIR", "#FFF3CD"),
        (6.45, 0.35, "ASK HUMAN", "#DDEBF7"),
        (9.65, 0.35, "ABSTAIN", "#FADBD8"),
    ]
    for x, y, label, color in action_specs:
        box(x, y, 1.9, 0.75, label, color)
        arrow((6.0, 2.05), (x + 0.95, y + 0.78))

    axis.text(
        6.0,
        8.20,
        "HumanGate-Q Risk-Adaptive Human-Agent Architecture",
        ha="center",
        va="top",
        fontsize=17,
        weight="bold",
        color="#263238",
    )
    _save(path)


def plot_class_distribution(target: pd.Series, path: Path) -> None:
    _style()
    order = [label for label in ("HIGH", "MEDIUM", "LOW") if label in set(target)]
    counts = target.value_counts().reindex(order)
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette=[PALETTE["teal"], PALETTE["gold"], PALETTE["red"]][: len(order)])
    ax.set(title="Kaggle Dataset Reliability-Class Distribution", xlabel="Reliability class", ylabel="Circuits")
    for patch, value in zip(ax.patches, counts.values):
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(), f"{int(value):,}", ha="center", va="bottom")
    _save(path)


def plot_confusion(matrix: np.ndarray, classes: list[str], path: Path) -> None:
    _style()
    plt.figure(figsize=(6.0, 5.0))
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    ax.set(title="Reliability Classifier Confusion Matrix", xlabel="Predicted", ylabel="True")
    _save(path)


def plot_reliability_diagram(
    y_true: pd.Series,
    probabilities: pd.DataFrame,
    classes: list[str],
    path: Path,
    bins: int = 10,
) -> None:
    _style()
    values = probabilities.to_numpy()
    confidence = values.max(axis=1)
    prediction = values.argmax(axis=1)
    true_index = np.array([classes.index(str(label)) for label in y_true])
    correct = prediction == true_index
    boundaries = np.linspace(0, 1, bins + 1)
    centers, accuracy, mean_confidence = [], [], []
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            centers.append((lower + upper) / 2)
            accuracy.append(float(correct[mask].mean()))
            mean_confidence.append(float(confidence[mask].mean()))
    plt.figure(figsize=(6.2, 5.2))
    plt.plot([0, 1], [0, 1], "--", color=PALETTE["gray"], label="Perfect calibration")
    plt.plot(mean_confidence, accuracy, marker="o", linewidth=2, color=PALETTE["blue"], label="HumanGate-Q reliability model")
    plt.xlabel("Mean calibrated confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Probability Reliability Diagram")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=True)
    _save(path)


def plot_policy_comparison(metrics: pd.DataFrame, path: Path) -> None:
    _style()
    selected = metrics[["unsafe_execution_rate", "human_review_rate"]].reset_index()
    long = selected.melt(id_vars="policy", var_name="metric", value_name="rate")
    labels = {
        "unsafe_execution_rate": "Unsafe execution",
        "human_review_rate": "Human review",
    }
    long["metric"] = long["metric"].map(labels)
    plt.figure(figsize=(10.0, 5.4))
    ax = sns.barplot(data=long, x="policy", y="rate", hue="metric", palette=[PALETTE["red"], PALETTE["blue"]])
    ax.set(title="Safety-Workload Comparison of Collaboration Policies", xlabel="Policy", ylabel="Rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="")
    _save(path)


def plot_action_distribution(actions: pd.DataFrame, path: Path) -> None:
    _style()
    counts = actions.apply(lambda column: column.value_counts(normalize=True)).T.reindex(columns=ACTIONS, fill_value=0.0)
    colors = [PALETTE["teal"], PALETTE["gold"], PALETTE["blue"], PALETTE["red"]]
    counts.plot(kind="barh", stacked=True, figsize=(9.0, 5.2), color=colors)
    plt.xlabel("Fraction of workflows")
    plt.ylabel("Policy")
    plt.title("Policy Action Distribution")
    plt.legend(title="Action", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(path)


def plot_risk_by_scenario(assessment: pd.DataFrame, path: Path) -> None:
    _style()
    order = (
        assessment.groupby("scenario")["risk_score"].median().sort_values().index.tolist()
    )
    plt.figure(figsize=(9.0, 5.2))
    ax = sns.boxplot(data=assessment, x="scenario", y="risk_score", order=order, color=PALETTE["blue"], showfliers=False)
    ax.set(title="HumanGate-Q Risk across Controlled Workflow Conditions", xlabel="Condition", ylabel="Fused risk score")
    ax.tick_params(axis="x", rotation=20)
    _save(path)


def plot_selective_curve(curve: pd.DataFrame, path: Path) -> None:
    _style()
    plt.figure(figsize=(6.5, 5.2))
    plt.plot(curve["automation_coverage"], curve["selective_risk"], marker="o", markersize=3, color=PALETTE["navy"], linewidth=2)
    plt.xlabel("Autonomous coverage")
    plt.ylabel("Unsafe fraction among automated cases")
    plt.title("Risk-Coverage Trade-off")
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    _save(path)


def plot_feature_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    _style()
    data = importance.head(top_n).sort_values("importance")
    plt.figure(figsize=(8.0, 6.0))
    ax = sns.barplot(data=data, x="importance", y="feature", color=PALETTE["teal"])
    ax.set(
        title=f"Top {len(data)} Pre-Execution Reliability Features",
        xlabel="Normalized model importance",
        ylabel="Feature",
    )
    _save(path)


def plot_ablation_heatmap(ablation_metrics: pd.DataFrame, path: Path) -> None:
    _style()
    columns = [
        "exact_action_accuracy",
        "unsafe_execution_rate",
        "human_review_rate",
        "appropriate_escalation_recall",
    ]
    data = ablation_metrics[columns].copy()
    data.columns = [
        "Action agreement",
        "Unsafe EXECUTE",
        "Human review",
        "Escalation recall",
    ]
    labels = {
        "Full HumanGate-Q": "Full HumanGate-Q",
        "Without reliability": "No reliability",
        "Without predictive uncertainty": "No predictive uncertainty",
        "Without structural shift": "No structural shift",
        "Without explicit shift flag": "No explicit shift flag",
        "Without both shift pathways": "No shift pathways",
        "Without verification": "No verification",
        "Without resource risk": "No resource risk",
        "Without criticality": "No criticality",
        "Without interactions": "No interactions",
    }
    data.index = [labels.get(index, index) for index in data.index]
    plt.figure(figsize=(8.2, 6.6))
    ax = sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.45,
        linecolor="white",
        cbar_kws={"label": "Proportion"},
    )
    ax.set(title="HumanGate-Q Component Ablation")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    _save(path)


def generate_all_figures(
    figure_directory: Path,
    full_target: pd.Series,
    model_result,
    assessment: pd.DataFrame,
    policy_metrics: pd.DataFrame,
    actions: pd.DataFrame,
    selective_curve: pd.DataFrame,
    ablation_metrics: pd.DataFrame,
    calibration_bins: int,
) -> list[Path]:
    paths = {
        "architecture": figure_directory / "figure_00_humangateq_architecture.png",
        "class_distribution": figure_directory / "figure_01_class_distribution.png",
        "confusion": figure_directory / "figure_02_confusion_matrix.png",
        "calibration": figure_directory / "figure_03_reliability_diagram.png",
        "policies": figure_directory / "figure_04_policy_safety_workload.png",
        "actions": figure_directory / "figure_05_action_distribution.png",
        "scenarios": figure_directory / "figure_06_risk_by_scenario.png",
        "selective": figure_directory / "figure_07_risk_coverage_curve.png",
        "importance": figure_directory / "figure_08_feature_importance.png",
        "ablation": figure_directory / "figure_09_ablation_heatmap.png",
    }
    plot_architecture(paths["architecture"])
    plot_class_distribution(full_target, paths["class_distribution"])
    plot_confusion(model_result.confusion, model_result.bundle.classes, paths["confusion"])
    plot_reliability_diagram(
        model_result.y_test.loc[model_result.test_probabilities.index],
        model_result.test_probabilities,
        model_result.bundle.classes,
        paths["calibration"],
        bins=calibration_bins,
    )
    plot_policy_comparison(policy_metrics, paths["policies"])
    plot_action_distribution(actions, paths["actions"])
    plot_risk_by_scenario(assessment, paths["scenarios"])
    plot_selective_curve(selective_curve, paths["selective"])
    plot_feature_importance(model_result.feature_importance, paths["importance"])
    plot_ablation_heatmap(ablation_metrics, paths["ablation"])
    return list(paths.values())
