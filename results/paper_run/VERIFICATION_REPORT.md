# Clean-Rerun Verification Report

**Verification date:** 26 August 2026  
**Repository version:** 2.1.0  
**Verification data:** the same QUASAR source file identified by the archived
SHA-256 digest

The complete experiment was rerun from the clean repository source after the
expanded ablation implementation was integrated. The rerun used a separate
output directory; it did not overwrite the archived chapter evidence.

## Comparison outcome

| Check | Outcome |
|---|---|
| Dataset SHA-256 | Exact match |
| Selected candidate | Exact match: `LGB-S15-C40` |
| Cross-validation accuracy, balanced accuracy, and macro-F1 | Exact match |
| Model metrics (16 recorded values) | Exact match |
| Selected policy thresholds | Exact match |
| Policy metrics for all six policies | Exact match |
| HumanGate-Q action counts | Exact match: 557 / 724 / 717 / 3,002 |
| Expanded 10-variant ablation metrics | Exact match for all common metrics |

The largest absolute difference in the model comparison was
`6.66 × 10⁻¹⁶`, which is ordinary floating-point representation noise.
Cross-validation fit and search durations differed because they are hardware-
and environment-dependent; the selected candidate and all cross-validation
scores were unchanged.

The original archived runtime was 20.84 seconds. The clean verification run
took 29.70 seconds in a different Linux/Python environment. Runtime is not a
scientific acceptance criterion.

## Automated verification

`python verify_package.py` passed dependency checks and all source and artifact
tests. The tests independently assert the headline metrics, selected model,
dataset provenance, action distribution, expanded ablation labels, 600-dpi
figure checksums, and absence of redistributed raw data.
