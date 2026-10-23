# Post review comparative analysis

Recorded 22 October 2026 after historical primary and DeepSeek outcomes were known,
before new Qwen participant inference and before three-validator Stack outcomes.
This is not independent preregistration or untouched confirmatory testing.

The scientific workflow remains observed CGM/HR → availability-aware preprocessing
→ participant-held-out forecasting → RF TreeSHAP → optional local LLM → typed
validation and controlled rendering, with deterministic comparator/fallback.

## Artifact separation and data recovery

Keep revision_v2 and revision_v3 historical artifacts unchanged in the local
archive. Recover missing source records using the pinned acquisition scripts in a
new Spark workspace. Require the original prepared array hashes:

- DDATSHR cbe727f2edf45ab0145774e03f235f34d2fa7c13c82d49c340c98a3d0458c7d9
- DiaData 54a12bfb016944a999ef51be91f749ba48c51b95c895b3a3d6bea1f893a4a35d

Rebuilt RF seed-42 fits go to revision_v4/recovered_rf_seed42, not historical fit
directories. Verify origin counts and every participant/horizon MAE against the
historical seed-42 table within 1e-7 mg/dL before generating any language cases.
Use the original 135-case manifest and original development example IDs. Check
selection-file equality with the frozen historical selection. Rebuilt checkpoints
are new files even if forecasts match; do not claim the original binary files
were recovered.

## Local Qwen factorial comparison

Eight LLM arms cross zero/few shot, calculated/supplied direction and plain/strict
JSON-schema decoding. The ninth arm deterministically supplies the expected
contract. Keep cases, numerical rounding, tier rules, validator, rendering,
temperature 0, seed 2026 and maximum output 2048 fixed. The schema constrains
structure/types and categorical domains; it must not encode the correct numeric
answers. Few-shot examples come exclusively from the historical development
people. Direction is computed from rounded contract values, matching the validator.

Development is 24 cases from 8 people; final test is 111 cases from 38 other
people. Require complete development integration before test. Freeze any choices
before test. Retain all failed requests, empty/truncated outputs, mismatches and
fallback origins in their original arm denominators. Do not silently change a
failed schema request to unconstrained generation or retry failed scientific
requests as if the initial attempt did not occur. Resume only already identified
artifacts under an identical configuration; changed code/config requires a new
output directory.

Identify the local model by the complete weight/metadata SHA-256 manifest and
record the served alias, model card, configuration architecture, quantization,
vLLM/PyTorch/container versions, context/KV settings and speculative-decoding
configuration. Local hashes identify the evaluated files but do not establish an
unrecorded upstream revision. Preserve the difference between the advertised
Qwen alias and internal architecture naming. Historical DeepSeek measurements
are a separate experiment and are never pooled with Qwen.

A technical nonparticipant schema-conflict probe precedes participant inference.
Record server backend configuration and empirical enforcement. API acceptance
alone is insufficient. No clinical or participant outcome is attributed to the
technical probe.

Requests are sequential, with case-specific rotated arm order. Run inference
after forecasting has stopped. Record pre/post server load, available memory,
request start/time, raw response, output tokens, finish reason, error/status body,
validation/rendering time and total contract-processing time. The model is already
loaded; no cold-start measurement is inferred. Report the first request per arm
separately from the remaining requests to expose compilation/warm-up effects.
These measurements exclude RF/SHAP recomputation, data acquisition and live alert
delivery, and differ from the historical end-to-end warm replay. Other clients or
server prefix caching limit isolation and must be disclosed.

Primary descriptive generation outcome: whole-contract acceptance, both case
fraction and equal-participant fraction with 10,000 participant-bootstrap resamples
(seed 917). Secondary outcomes: field-level errors, parse/omission/extra-field
errors, direction/tier errors, API errors, token counts and timing distributions.
Individual-field fractions use all expected leaves as the denominator; missing
fields cannot disappear from the denominator.

Three post-review exploratory factor contrasts average over the other two factors:
supplied minus calculated direction; schema minus plain decoding; few minus zero
shot. Calculate paired differences within each participant, bootstrap participants,
and use two-sided Pratt Wilcoxon tests with a separate three-test Holm family.
Report all eight arms even when a main effect masks interactions. These analyses
do not extend the historical primary 18-test family or prove general superiority.

## Full Stack sensitivity

Use unchanged prepared arrays/origins and the existing deterministic three-person
validation allocation. Fit RF, XGBoost, TFT-inspired encoder, BiLSTM–Transformer
and CNN–GRU in every outer fold for both cohorts with CGM/HR and seeds 42, 1729,
2026: 46 people × 5 members × 3 seeds = 690 base fits. The CGM-only neural extension
is analyzed from existing matched primary summaries; no unperformed second full
feature-condition sensitivity is claimed.

Use the existing hyperparameter budget, maximum 80 epochs, patience 10,
training-only scaling, equal-participant base-training loss and equal-participant
validation MAE. Fit nonnegative sum-to-one weights separately for each horizon,
fold and seed using equal-participant validation squared error. Never fit weights,
select configurations or stop training using the held-out participant. Verify
identical test sample IDs and targets across all members. Preserve checkpoints,
histories, search tables, validation/test predictions, weights and manifests.

Benchmark one predetermined fold (first sorted DDATSHR ID, seed 42). The pilot
does not select a new hyperparameter budget based on test performance. Report
environment changes from the historical H200 setup. Training and LLM request
phases remain separate; retain at least 8 GiB available memory and stop the new
job rather than stopping unrelated services if that guard fails.

Report participant-macro MAE for seed-averaged predictions and paired descriptive
intervals for Stack minus RF and three-validator minus one-validator scores.
The validation change also reduces base-training population size. Do not invent
an MCID, infer clinical benefit from statistical significance, or count seeds/
windows as independent people. Preserve negative results.

## Manuscript and author boundaries

Remove the short-context Chronos comparison, as permitted by the review, while
retaining superseded artifacts. Reframe title/abstract/contribution around actual
benchmark and output-consistency evidence. Keep human comprehension/usefulness,
clinical outcomes and live operation explicitly unevaluated. Author verification,
CRediT, institutional determinations and any prior-publication correction or
editorial correspondence require the responsible humans; no ratings, approval,
correction or communication is simulated.
