<!-- CURRENT_REVISION_20261023 -->
# Current review revision

The Qwen-primary review deliverables and reproducible supplement are in [revision_20261023](revision_20261023/DELIVERY_STATUS.md). Use its clean manuscript, selective-yellow copy and point-by-point response as the current revision. [Supplementary File S1](revision_20261023/supplement/README.md) includes code, source terms, frozen protocols, derived evidence and audit manifests.

All 888 Qwen test requests, 690 Stack fit manifests, 138 validation-only weight sets and 10 representative checkpoints were audited. The DOCX files were rendered and every page inspected. Funding is confirmed as no external funding. Human usefulness, historical structured DeepSeek evidence, raw historical neural HR predictions and the author decisions listed in DELIVERY_STATUS remain unresolved. No author approval or submission readiness is claimed.

The files and claims below belong to the earlier publication snapshot and are preserved as historical work; they are superseded for this review. No new reproduction of historical DeepSeek is asserted.

---

# New Paper — GlucoSense research revision

**Multimodal Deep Learning for Real-Time Glucose Prediction with Zero/Few-Shot On-Premises LLM Explainability: External Validation on an Independent Adolescent Cohort**

This repository contains the revised research manuscript, executable experiments, aggregate results, figures and reproducibility evidence. It preserves the original title and research goal. This is a retrospective computational study and a submission draft—not an accepted article or a clinical trial.

## Start here

- [Final manuscript — PDF](final/manuscript_REVISED_EVIDENCE.pdf)
- [Final manuscript — editable Word](final/manuscript_REVISED_EVIDENCE.docx)
- [Prepared response to feedback — Word](final/RESPONSE_TO_PROFESSOR_REVISED.docx) / [Markdown](final/RESPONSE_TO_PROFESSOR_REVISED.md)
- [Submission readiness and remaining author responsibilities](project/SUBMISSION_READINESS.md)
- [Reproduction instructions](project/REPRODUCIBILITY.md)
- [Verified supplementary archive](revision_evidence_supplement.zip) and [checksum](revision_evidence_supplement.zip.sha256)
- [Public-release contents and exclusions](PUBLICATION_CONTENTS.md)

## Research and evidence

The primary evaluation includes 17 DDATSHR participants and 29 non-overlapping-source DiaData participants, with observed targets, patient-disjoint model selection, three seeds, matched baselines and ablations. All 1,656 primary fits completed. The primary statistical family contains 18 Holm-adjusted comparisons. Stacking improves RF on DDATSHR; HR addition has no demonstrated significant benefit. Negative findings are retained.

Additional experiments include 18 development fits frozen before AZT1D evaluation (24 participants, 92,976 origins), 276 RF sensitivity fits with three validation participants, a pinned Chronos-2 baseline on unchanged primary test origins, and 90 sequential warm processing measurements. The original adolescent evaluation uses retraining; the additional frozen-model experiment is glucose-only transfer to AZT1D. Neither verifies unavailable historical multimodal weights.

The LLM study uses actual held-out RF forecasts and additive TreeSHAP. Its 111 test cases come from 38 people separate from development participants. Zero-shot and few-shot contract acceptance is 100/111 and 96/111, versus 111/111 for deterministic rendering. Source-relative consistency is not proof of clinical safety or correct source measurements. Warm replay excludes acquisition, startup and clinical workflow.

## Repository layout

| Location | Contents |
| --- | --- |
| `final/` | Convenient manuscript and prepared response copies |
| `project/src/revision/` | Data preparation, training, analysis and output validation |
| `project/scripts/` | Acquisition, experiment, verification, plotting and document builders |
| `project/tests/` | Ten regression tests |
| `project/results/revision_v2/` | Primary aggregate results, figures, prompts and provenance |
| `project/results/revision_v3/` | Additional external, baseline, robustness and latency evidence |
| `project/manuscript/` | Evidence-linked manuscript source and template |
| `project/protocol_v2.md`, `project/protocol_v3.md` | Recorded protocols and limitations |

The supplementary extraction preserves audit copies of fit manifests under `provenance/fit_manifests`, rather than pretending that missing large checkpoints are completed runnable caches. Follow the reproduction instructions to acquire source releases and regenerate large artifacts. The exact NVIDIA/Python environments and separate local LLM environment are documented there.

## Verification and publication status

All 18 additional external checkpoints reproduced saved predictions on reload. More than 1.1 million external input/target boundaries were checked against raw observations. Chronos-2 horizon alignment and inference-context independence were checked. Ten tests passed both in the research workspace and in a fresh supplement extraction. Seven figures, eight manuscript tables and 19 references are supplied.

Responsible authors must still approve the full research, finalize declarations, confirm applicable secondary-data requirements and resolve disclosure of the related prior publication. See [the prepared disclosure](project/PRIOR_PUBLICATION_DISCLOSURE.md). Integrity checks do not replace peer review or guarantee acceptance.

Raw participant records, exact generation inputs/outputs, model weights, credentials and original private professor correspondence are not redistributed. The prepared response is the authors’ draft response, not the professor’s original correspondence. Source datasets and models retain their own licenses and attribution requirements; this release does not assign a new blanket license to third-party material.

Public release repository: https://github.com/hossain0707/new-paper. The original research snapshot and supplement retain their historical provenance and original repository references. Refer to PUBLIC_RELEASE_CHECK.json for the local packaging audit.
