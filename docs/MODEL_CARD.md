# Model Card: HumanGate-Q Reliability Agent

## Model details

- **Task:** Three-class pre-execution circuit-reliability classification
- **Backend:** LightGBM multiclass classifier
- **Selected candidate:** `LGB-S15-C40`
- **Calibration:** Scalar temperature scaling on a separate 2,400-row partition
- **Features:** 28 leakage-safe numeric circuit-structure features
- **Seed:** 42

Selected parameters are 220 estimators, learning rate 0.04, 15 leaves, minimum
child size 40, L2 regularization 1.0, and full row and column sampling. Selection
used three-fold stratified training-only cross-validation.

## Held-out performance

| Metric | Value |
|---|---:|
| Accuracy | 0.895500 |
| Balanced accuracy | 0.876599 |
| Macro-F1 | 0.875389 |
| Log loss | 0.240996 |
| Multiclass Brier score | 0.146694 |
| Expected calibration error | 0.007449 |
| Temperature | 0.956886 |

The same-split calibrated ExtraTrees baseline achieved 0.890333 accuracy. The
LightGBM improvement in accuracy is modest; balanced accuracy decreased
slightly. The clearest improvement is in the probability-sensitive metrics.

## Intended role

The classifier is one signal inside the HumanGate-Q governance policy. It is not
a standalone safety decision. Its probabilities are combined with uncertainty,
shift, verification, resource, and criticality evidence.

## Limitations

- Labels are derived from simulated-noise behavior, not live QPUs.
- Feature importance is dataset-specific and is not causal evidence.
- Aggregate ECE can hide class-conditional or local calibration errors.
- The model has not been externally validated across devices, datasets, or
  institutions.
- A high confidence score must not bypass verification or governance controls.
