# Archived Chapter Run

This directory is the immutable evidence package for the results reported in
the HumanGate-Q chapter. Fresh executions belong under `outputs/`.

The browser-friendly plots are in `figures/`; the separate 600-dpi submission
masters and their checksums are documented in
[`PUBLICATION_FIGURES.md`](PUBLICATION_FIGURES.md).

## Figure mapping

| Repository file | Chapter figure |
|---|---|
| `figures/figure_00_humangateq_architecture.png` | Figure 14.1 — architecture |
| `figures/figure_01_class_distribution.png` | Figure 14.2 — class distribution |
| `figures/figure_02_confusion_matrix.png` | Figure 14.3 — confusion matrix |
| `figures/figure_03_reliability_diagram.png` | Figure 14.4 — reliability diagram |
| `figures/figure_04_policy_safety_workload.png` | Figure 14.5 — safety/workload comparison |
| `figures/figure_05_action_distribution.png` | Figure 14.6 — action distribution |
| `figures/figure_07_risk_coverage_curve.png` | Figure 14.7 — risk–coverage curve |
| `figures/figure_06_risk_by_scenario.png` | Figure 14.8 — risk by scenario |
| `figures/figure_08_feature_importance.png` | Figure 14.9 — feature importance |
| `figures/figure_09_ablation_heatmap.png` | Figure 14.10 — expanded ablation heat map |

## Provenance

- Generated from the Kaggle-only dataset identified in the root README.
- Random seed: 42.
- Source data SHA-256:
  `506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c`.
- Original end-to-end runtime: 20.838636 seconds.
- The full effective configuration and package versions are in
  `run_metadata.json`.

The expanded ablation table and heat map incorporate the final analysis that
separates predictive uncertainty, structural shift, the explicit verifier
shift flag, both shift pathways, and resource risk.
