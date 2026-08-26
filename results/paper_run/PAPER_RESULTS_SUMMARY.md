# HumanGate-Q Verified Results Summary

> This file is generated from the completed experiment. Use these values—not estimates or unit-test output—in the chapter.

## Dataset and split

- Kaggle rows used: **30,000**
- Leakage-safe structural features: **28**
- Controlled held-out workflow cases: **5,000**
- Dataset SHA-256: `506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c`

## Reliability model

- Training-only cross-validation selected: **LGB-S15-C40**
- Accuracy: **89.55%**
- Balanced accuracy: **87.66%**
- Macro F1: **0.8754**
- Expected calibration error: **0.0074**
- Multiclass Brier score: **0.1467**
- Learned calibration temperature: **0.9569**
- Original Extra Trees accuracy on the identical test split: **89.03%**
- Absolute accuracy change: **+0.52 percentage points**

## Primary policy results

- Policy-threshold validation cases (separate from test): **2,400**
- HumanGate-Q exact action accuracy: **74.40%**
- HumanGate-Q unsafe workflow-execution rate: **9.69%**
- HumanGate-Q appropriate escalation recall: **87.63%**
- HumanGate-Q safe automation coverage: **90.04%**
- HumanGate-Q human-review rate: **14.34%**
- Absolute unsafe-execution reduction versus full autonomy: **73.25 percentage points**
- Absolute human-review reduction versus always-human review: **85.66 percentage points**

## Ablation interpretation

The detailed ablation table is in `tables/ablation_metrics.csv`. Compare every variant with `Full HumanGate-Q`; do not claim that a component helps unless its removal worsens the relevant metric in this run.

## Runtime

- End-to-end experiment runtime: **20.84 seconds**

## Files for the chapter

- `tables/policy_metrics.csv` — main comparison
- `tables/model_selection_cv.csv` — training-only candidate selection
- `tables/model_comparison.csv` — original versus upgraded test metrics
- `tables/policy_threshold_search.csv` — validation-only safety-constrained threshold selection
- `tables/bootstrap_confidence_intervals.csv` — 95% intervals
- `tables/ablation_metrics.csv` — component study
- `tables/workflow_decisions.csv` — auditable per-case decisions
- `figures/` — ten high-resolution figures, including the system architecture

## Required limitation statement

The experiment uses synthetic circuits and simulated-noise reliability labels from Kaggle. Workflow criticality and failure conditions are controlled perturbations, and the reference human-oversight action is a declared experimental oracle rather than observed human behaviour. Results therefore demonstrate policy behaviour in a reproducible simulation and do not establish real-QPU or real-world safety.
