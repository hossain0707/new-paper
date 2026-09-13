# GlucoSense AI: corrected original paper

The exact original title, GlucoSense goal and professor-reviewed section sequence are retained; incorrect results are corrected from verified primary and additional evidence.

Use [the revised manuscript](manuscript/manuscript_REVISED_EVIDENCE.md), [submission readiness](SUBMISSION_READINESS.md), and [reproduction commands](REPRODUCIBILITY.md). Current tables and figures are under `results/revision_v2/` and `results/revision_v3/`; historical `results/tables/` and older drafts are superseded.

The corrected experiment uses observed targets, sensor-aware 15-minute availability conventions, identical model test manifests, patient-disjoint validation, three seeds and directly constrained stacking. All 1,656 primary fits completed. Checkpoints, per-origin predictions and real generation archives remain on the local resource. The supplementary ZIP at the repository root includes reproducible code, aggregate results, configuration manifests, provenance and all seven figures.

RF subject-macro MAE at 30/60/90 minutes is 18.12/30.54/38.82 mg/dL for DDATSHR and 23.92/34.25/41.63 mg/dL for DiaData. The stack's adjusted improvement over RF occurs on DDATSHR; HR addition is not significant in either cohort. The experiment does not verify the historical six-modal OhioT1DM claims or demonstrate clinical benefit.

The local generation study has 24 development cases from eight people and 111 test cases from 38 other people, with deterministic, zero-shot and few-shot conditions. Results include raw contract failures, source-corruption boundary tests and patient-cluster acceptance intervals. Source-relative consistency is not source-truth verification.

Additional evidence includes frozen DDATSHR-to-AZT1D glucose-only transfer (24 participants), 276 three-validator RF sensitivity fits, a pinned Chronos-2 baseline on unchanged primary origins, and 90 sequential warm processing measurements. All 18 external checkpoints were reproduced on reload; 1,115,712 raw observation boundaries were independently checked. Ten regression tests pass. These are separately labelled extensions, not replacements for the primary significance family.

The professor response and highlighted manuscript are prepared privately under `professor feedback/` and have not been sent. Human author declarations remain pending. See the readiness assessment for exact remaining requirements.
