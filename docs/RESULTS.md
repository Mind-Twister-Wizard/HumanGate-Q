# Results and Evidence Index

The values below are copied from the immutable archived run. Machine-readable
sources are under [`results/paper_run`](../results/paper_run/).

## Reliability model

| Metric | LightGBM | ExtraTrees baseline |
|---|---:|---:|
| Accuracy | 89.55% | 89.03% |
| Balanced accuracy | 87.66% | 87.74% |
| Macro-F1 | 87.54% | 87.19% |
| Log loss | 0.2410 | 0.2538 |
| Multiclass Brier | 0.1467 | 0.1532 |
| ECE | 0.74% | 1.02% |

## HumanGate-Q policy

| Metric | Value | Denominator |
|---|---:|---|
| Reference-action agreement | 74.40% | All 5,000 workflows |
| Unsafe `EXECUTE` | 9.69% | Workflows assigned `EXECUTE` |
| Escalation recall | 87.63% | Workflows requiring `ASK_HUMAN` or `ABSTAIN` |
| Safe automation coverage | 90.04% | Workflows for which automation was reference-safe |
| Human review | 14.34% | All workflows |

Action counts were 557 `EXECUTE`, 724 `SELF_REPAIR`, 717 `ASK_HUMAN`, and
3,002 `ABSTAIN`.

## Where each claim can be audited

| Evidence | File |
|---|---|
| Model scores and split sizes | `tables/model_metrics.csv` |
| Same-split model comparison | `tables/model_comparison.csv` |
| Cross-validation selection | `tables/model_selection_cv.csv` |
| Policy-threshold selection | `tables/policy_threshold_search.csv` |
| Policy and baseline metrics | `tables/policy_metrics.csv` |
| Bootstrap intervals | `tables/bootstrap_confidence_intervals.csv` |
| Expanded component ablation | `tables/ablation_metrics.csv` |
| Coefficient sensitivity | `tables/coefficient_sensitivity.json` |
| Per-workflow evidence and actions | `tables/workflow_decisions.csv` |
| Effective configuration and environment | `run_metadata.json` |
| Fitted calibrated model | `models/humangateq_reliability_model.joblib` |

## Interpretation boundary

These results measure agreement and risk/workload properties relative to a
declared controlled reference policy. Because the circuits, noise-derived
labels, scenarios, and oracle are synthetic or constructed, the values are not
real-world safety certification.
