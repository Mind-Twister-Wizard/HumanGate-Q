# Dataset Card: QUASAR

## Source

- **Name:** Synthetic Quantum Circuit Reliability Dataset (QUASAR)
- **Kaggle page:** <https://www.kaggle.com/datasets/ykmadhav/synthetic-quantum-circuit-reliability-dataset>
- **Kaggle slug:** `ykmadhav/synthetic-quantum-circuit-reliability-dataset`
- **Chapter-run access date:** 20 August 2026
- **Chapter-run source SHA-256:**
  `506eff548af0603f771290cb75d3515f203d976162f3171b6494a1bb4d94025c`
- **License:** Follow the terms supplied on the Kaggle dataset page and in the
  downloaded archive.

HumanGate-Q does not redistribute the dataset. `download_dataset.py` obtains it
from Kaggle, and `.gitignore` excludes downloaded files and credentials.

## Target and sample

The required target is `reliability_class` with `HIGH`, `MEDIUM`, and `LOW`
labels. The paper run uses a seed-42 stratified sample of 30,000 circuits.
Class counts are 7,613 `HIGH`, 14,999 `LOW`, and 7,388 `MEDIUM`.

## Excluded columns

The implementation predicts reliability before execution and therefore drops:

- identifiers such as `circuit_name` and `source_file`;
- `reliability_class` and `reliability_score`; and
- `estimated_fidelity`, distance measures, and ideal/noisy success
  probabilities that can reveal the simulation-derived target.

The exact list removed in the paper run is stored in
`results/paper_run/run_metadata.json`.

## Derived scenarios

The five workflow conditions are deterministic controlled perturbations of
held-out Kaggle rows, not additional datasets. Metadata missingness and
structural shift are injected after model fitting. Criticality values represent
experimental consequence levels rather than observed application provenance.

## Appropriate and inappropriate use

The data support controlled research on reliability classification,
uncertainty, distribution shift, and oversight allocation. They do not support
claims about live hardware, human behavior, device-specific guarantees, or
clinical, financial, or cryptographic deployment safety.
