# Analysis protocol v2

Written before the corrected forecasting experiments (12 September 2026).
This is a time-stamped analysis plan, not a registered protocol or independent preregistration.

## Questions and estimands
H1: Does adding observed heart rate improve subject-average forecast error over CGM-only
on a matched sample? H2: Do deep learners or a constrained ensemble outperform persistence
and tuned tree baselines under the same cohort/window protocol? H3: Do explicit rules and
typed, field-aware validation reduce detectable narrative factual errors compared with an
unguarded generator, and what coverage is sacrificed? Outcomes can be negative.

Primary forecasting outcome: subject-macro MAE at 30, 60, 90 minutes; secondary pooled and
subject-macro RMSE, full Clarke zones, by-range error and observed low/high target classification sensitivity/precision (not prospective episode alerts).
Subject, not overlapping row, is the inferential unit. Report paired effect sizes,
patient-cluster bootstrap intervals and Holm-adjusted two-sided patient-level tests for
prespecified primary comparisons (RF CGM+HR vs RF CGM-only; ensemble vs RF; RF vs persistence,
at each horizon), with one Holm family across both datasets (18 tests). Small cohorts remain an explicit uncertainty limitation.

## Data
DDATSHR original Abbott and Medtronic CGM with Fitbit HR; DiaData SDBII RAW, with all
DDATSHR-source subjects removed. Retain HUPA-UCM and D1NAMO source labels. No claim that
D1NAMO is independent of every dataset used in the previous publication. No use of the
previous six-modal model weights or previously imputed streams.

Use right-labelled/right-closed 15-minute bins. At each boundary choose the most recent
observed CGM in that bin and mean observed HR. No interpolation of CGM, HR or targets.
Timestamp inputs must carry offsets when present. Exclude the known ambiguous DST interval
and uncorrectable clock intervals, logging counts and decisions. Never infer physiology
from grid missingness alone. All features available by prediction time; outcomes are
observed CGM values in bins 30/60/90 minutes later (report discretization convention).

DiaData-specific availability safeguard (added before its corrected model runs): upstream
integration rounds timestamps and the 15-minute export does not preserve original reading
times. Delay the entire input sequence by one 15-minute step relative to the supplied
forecast-origin label; do not delay targets. Thus the latest input is labelled t−15 minutes
and the outcomes t+30/60/90. This prevents use of a contemporaneous rounded future reading
under a within-bin aggregation assumption, but cannot establish exact source-clock accuracy.
Report DiaData as a delayed-input retrospective replication, not exact real-time validation.
DDATSHR uses original offset-bearing timestamps and needs no extra input delay.

Primary shared input: 2-hour history (9 boundaries including current time) of glucose and
HR plus known time-of-day sin/cos. Complete observed inputs and all three targets required;
at least 50 shared examples per evaluable patient. Record pre/post exclusions by patient,
device and source. CGM-only ablation uses exactly the same examples, with HR removed.
Secondary 4-hour history sensitivity is analysed separately, never substituted for the
primary result on the basis of test performance.

## Model selection and estimation
Outer leave-one-patient-out within each cohort. For each outer fold, deterministically
select an inner-validation patient from training patients using a seeded shuffle. Keep this
patient entirely outside base-model fitting; use inner predictions for model selection,
stack weights, and descriptive residual interval calibration. Outer test data are never
used for any of these choices. All learned models in a comparison use identical base-train,
validation, and test subjects; report this nested-holdout design rather than full nested CV.

Ridge/RF/XGBoost: small explicit grids in the saved configuration. Deep models: learning
rates 1e-3 and 3e-4, max 80 epochs, patience 10, restore lowest subject-validation MAE
checkpoint. Predict standardized glucose changes from current CGM, add current CGM back
for evaluation. Training-only feature and target normalization. Mini-batches contain 1024 windows. Training losses and tree fits use inverse-patient-window-count
weights (normalized to mean one), so each base-training patient has equal total weight.
Deterministic PyTorch algorithms and math attention are enabled; concurrent processes share
the four GPUs, so recorded wall times are throughput measurements, not isolated latency.
Three fixed seeds
(42, 1729, 2026); report seed variation and a seed-averaged prediction alongside per-seed
results. Architecture variants must be named accurately (TFT-inspired encoder).

Stack five members using nonnegative weights constrained to sum to one, optimized directly
on inner-validation predictions. Save each member prediction. Tree-only ensemble is an
ablation. A limited validation patient creates uncertain stacking and calibration estimates;
report those limits. Secondary source-held-out transfer fits RF and XGBoost with three seeds in both
HUPA-UCM→D1NAMO and D1NAMO→HUPA-UCM directions, reserving one training-source patient
for validation. Four-hour history sensitivity fits Ridge and XGBoost with seed 42.
For history sensitivity, refit both the two- and four-hour models on the four-hour eligible
patient/timestamp manifest, ensuring identical training/validation/test allocations.
History comparisons must be restricted to patient/timestamps present in both histories;
report changing eligibility separately. These secondary analyses are exploratory.

## Explanations
Archive per-case held-out RF predictions and genuine TreeSHAP values. State which model
is explained. No evidence that explanations establish causality. Explicit units, time
horizons, and complete study-specific score-to-tier rules; tier rules are an experimental
consistency task, not a validated clinical triage protocol. Avoid medication/dose advice
and claims that alerts were dispatched. Do not claim clinical acceptability without human
clinical review. Do not claim equivalence to fine-tuning without a paired comparison.

Select cases across patients, sources and glycemic states using fixed seeds, with a
development/test split by patient. Use deterministic template, zero-shot and few-shot
local-generation conditions. Freeze prompt and detector before final test generation.
Report model identity/revision, server settings, prompts, outputs, refusals, failures,
claim coverage and latency. Detector tests use a separately defined mutation suite, with
held-out mutation families for challenge evaluation; machine-generated gold is not
clinician annotation. Preserve false negatives and do not silently revise on final tests.

## Reporting
All new results/plots must derive from versioned predictions. Unsupported historical
results and reused architecture art will not be presented as new evidence. Include a
prior-publication difference statement and correct literature citations. No claim of
statistical significance or causality based on uncorrected exploratory comparisons.

## Reporting clarification (13 September 2026, after model fitting)
The synthetic challenge families are distinct from the development mutation families, but
the same implementation process designed and inspected both. Accordingly, the paper
calls them separately designated challenge families, not a blinded independent adversarial
validation. This clarification changes interpretation, not the frozen detector or test outputs.
Participant-cluster descriptive intervals and rejection categories were added to final
generation reporting without changing the cases, prompts, model or acceptance rule.
