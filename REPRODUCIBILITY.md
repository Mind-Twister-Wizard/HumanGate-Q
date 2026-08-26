# Reproducibility Protocol

This document distinguishes the immutable chapter evidence from new local
runs. `results/paper_run/` contains the audited outputs reported in the
manuscript. `outputs/` is reserved for reruns and is ignored by Git.

## Exact experimental design

| Item | Archived chapter run |
|---|---|
| Random seed | 42 |
| Kaggle rows used | 30,000 |
| Leakage-safe features | 28 |
| Training rows | 19,200 |
| Calibration rows | 2,400 |
| Policy-validation rows | 2,400 |
| Test rows | 6,000 |
| Controlled test workflows | 5,000 |
| Model-selection candidates | 6 LightGBM configurations |
| Cross-validation | 3-fold stratified, training-only |
| Threshold candidates | 540, validation-only |
| Bootstrap repetitions | 300 paired resamples |

The selected candidate was `LGB-S15-C40`: 220 trees, learning rate 0.04,
15 leaves, minimum child size 40, L2 regularization 1.0, and full row/column
sampling. Temperature scaling produced a temperature of 0.956886.

## Data provenance check

The source CSV used in the paper run had SHA-256:

```text
506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c
```

`python verify_package.py` reports whether a locally downloaded source matches
this digest. A different digest does not prove that the file is invalid, but it
means exact numerical reproduction is no longer guaranteed.

## Full reproduction

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python download_dataset.py
python verify_package.py
python run_all.py
python compare_to_paper_run.py
```

The run performs, in order:

1. data discovery, schema validation, stratified row limiting, and leakage
   removal;
2. four-way train/calibration/policy-validation/test partitioning;
3. training-only LightGBM selection and same-split ExtraTrees comparison;
4. held-out probability calibration;
5. validation-only policy threshold selection;
6. deterministic scenario creation and five-channel risk assessment;
7. HumanGate-Q and five baseline policies;
8. paired bootstrap, selective-risk, and 10-variant ablation analyses; and
9. model, table, figure, decision, and metadata export.

## Acceptance checks

For the original data version and compatible library versions, the headline
values should match `results/paper_run/` to ordinary floating-point tolerance:

- accuracy: 0.8955;
- macro-F1: approximately 0.875389;
- ECE: approximately 0.007449;
- action agreement: 0.7440;
- unsafe `EXECUTE` rate: approximately 0.096948;
- escalation recall: approximately 0.876296;
- safe automation coverage: approximately 0.900352; and
- human-review rate: 0.1434.

Run metadata record the original package versions and operating system. Minor
plot rasterization and timing differences across platforms are expected; the
decision outputs and reported metrics should remain stable.

## Reproducibility boundaries

The workflow scenarios, criticality assignments, and reference actions are
declared experimental constructs. Reproducing them verifies the implementation
and reported simulation, not deployment safety or human-subject behavior.
