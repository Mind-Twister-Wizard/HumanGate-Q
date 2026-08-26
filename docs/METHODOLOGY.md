# Methodology

## Objective and research questions

HumanGate-Q evaluates whether a risk-adaptive gate can reduce unsafe autonomous
quantum-workflow execution without imposing the 100% review demand of an
always-human policy.

- **RQ1:** Can the calibrated model classify pre-execution circuit reliability?
- **RQ2:** Does the full gate improve reference-action agreement and balance
  safety with human workload relative to simpler policies?
- **RQ3:** Which evidence channels influence the gate's decisions?
- **RQ4:** How does the policy behave under ambiguity, tool failure, missing
  metadata, and distribution shift?

## Data preparation and leakage control

The experiment draws exclusively from the Kaggle QUASAR dataset. The loader
normalizes `reliability_class`, uses a seed-42 stratified sample of 30,000 rows,
and removes identifiers plus simulation-derived variables that reveal or
closely determine the target. Twenty-eight numeric structural features remain.

Rows are divided into mutually exclusive partitions: 19,200 for training,
2,400 for temperature calibration, 2,400 for policy-threshold validation, and
6,000 for final testing. Test labels are not used for model selection,
calibration, or threshold selection.

## Reliability model

Six compact LightGBM candidates are compared using three-fold stratified
cross-validation restricted to the training partition. The winning candidate
is fitted to the training data. An ExtraTrees model is fitted on the identical
split as a declared baseline. Scalar temperature scaling is learned only on the
calibration partition.

For multiclass expected calibration error (ECE), 10 equal-width confidence
bins are used with the top-label definition. Empty bins are omitted.

## Controlled workflows

Five thousand held-out test rows are assigned reproducibly to `clean`,
`ambiguous_goal`, `tool_failure`, `metadata_missing`, or
`distribution_shift`. Application-domain labels set controlled criticality
ranges; they do not imply that the Kaggle records originated in those domains.

## Risk fusion and action policy

The risk vector contains calibrated reliability risk, predictive entropy,
structural shift, verification risk, resource complexity, and workflow
criticality. The configured weighted score also includes
reliability–criticality and verification–criticality interactions and is
clipped to `[0, 1]`.

The gate applies `EXECUTE`, `SELF_REPAIR`, `ASK_HUMAN`, and `ABSTAIN` in a fixed
precedence order. Tool failure, a sufficiently high fused score, or a high
calibrated probability of `LOW` reliability forces abstention. Repeated repair
attempts are outside this single-pass experiment.

The configured threshold grid contains 540 ordered candidates. Selection uses
only policy-validation workflows and considers unsafe execution, escalation
recall, safe automation coverage, human review, and exact reference-action
agreement under declared constraints.

## Reference policy and baselines

The transparent experimental reference policy uses held-out reliability labels,
scenario flags, and criticality. It is not a learned human-behavior model or a
universal ethical policy.

HumanGate-Q is compared with Fully Autonomous, Always Human, Confidence Only,
Criticality Only, and Verifier Only policies. Metrics use explicit
denominators; in particular, unsafe `EXECUTE` is conditional on workflows
assigned `EXECUTE`, while human review is the share of all workflows assigned
`ASK_HUMAN`.

## Statistical and component analyses

The package produces 300 paired nonparametric bootstrap resamples for the main
policy metrics and a risk–coverage curve. Its 10 ablation conditions separate:

1. the full policy;
2. reliability;
3. predictive uncertainty;
4. structural distribution shift;
5. the explicit shift flag in verifier evidence;
6. both shift pathways;
7. verification;
8. resource risk;
9. criticality; and
10. interaction terms.

All thresholds remain fixed at the selected operating point during ablation.

## Audit trail

Each run exports the fitted model, model-selection table, threshold search,
calibration and policy metrics, all workflow assessments and actions, bootstrap
intervals, ablation outputs, plots, the effective configuration, package
versions, dataset hash, and runtime.
