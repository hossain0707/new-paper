# Submission readiness — strengthened original paper, 13 September 2026

**NOT READY FOR MDPI SUBMISSION** until the three critical author responsibilities below are resolved. The computational experiments and verification are complete. This is a retrospective computational study; no clinical trial is planned or claimed.

## Completed and verifiable

- Original title, GlucoSense research goal, nine main sections and results Sections 7.1–7.5 preserved.
- All 1,656 primary fits: matched observed targets, two feature conditions, three seeds, patient-disjoint selection, restored neural checkpoints and constrained ensembles. Primary results and the 18 Holm-adjusted tests remain unchanged.
- Source acquisition and clock/parser audits, source-overlap exclusion, complete Clarke zones, glycemic-range errors, participant-level uncertainty, history/source sensitivities and participant-907 error analysis.
- Actual local generation with genuine TreeSHAP, 111 patient-separated test cases, deterministic comparator, retained contract failures, synthetic challenges and source-corruption failures. No clinical-safety or source-truth guarantee is inferred.
- **New frozen-model transfer:** 18 DDATSHR development fits frozen before AZT1D scoring; 24 evaluable participants and 92,976 origins. Subject 14's ambiguous CGM/BGM field was excluded before scoring. All 1,115,712 input/target boundaries were independently matched to raw observations. This is glucose-only transfer to an AID cohort, not frozen historical multimodal validation in adolescents.
- **Validation robustness:** all 276 RF fits with three inner validation participants, preserving primary outer targets. The analysis changes both validation coverage and training size and does not replace the primary findings.
- **Modern baseline:** pinned Chronos-2 on every primary test origin, with horizon alignment and independence from other inference contexts checked. Short context, unknown pretraining overlap and the RF comparator's additional time-of-day features are disclosed.
- **Processing replay:** 90 measured runs over 30 cases from 24 people, plus three excluded warmups. Median complete warm processing: deterministic 0.124 s, zero-shot 26.600 s, few-shot 26.659 s. Acceptance: 30/30, 26/30 and 25/30 respectively. No observed 900-second deadline misses. Device acquisition, startup and clinical workflow are excluded; this is not live or hard real-time validation.
- All 18 external checkpoints reproduced saved predictions on reload. Primary and additional integrity checks pass; 10 regression tests pass. Seven figures, eight manuscript tables, 19 references, clean Word/PDF, private professor response and reproducible supplement are prepared.

Verification records: `results/revision_v2/artifact_verification.json`, `results/revision_v3/artifact_verification.json`, and the additional raw-boundary/checkpoint/Chronos audit files. Integrity checks are not independent scientific peer review.

## Critical — responsible authors must resolve before submission

1. **Review and approve the entire revised research.** Confirm authors/affiliations, CRediT roles, funding, conflicts, substantive AI-assistance disclosure and accountability. Replace the pending back-matter statements with truthful approved declarations. Automated work cannot supply author approval.
2. **Confirm institutional secondary-data and source-use requirements.** Obtain the institution's review/exemption determination where applicable and accurate consent/data-use wording. Public availability does not establish an institutional determination. This does not mean conducting a new clinical trial.
3. **Resolve the related prior publication transparently.** Review `PRIOR_PUBLICATION_DISCLOSURE.md`, supply the earlier article to the editor and address its unavailable original evidence and unsupported modality pathway with the responsible authors/publisher as appropriate. The new experiments cannot retroactively verify that article. The prepared cover-letter material discloses the relationship; it has not been sent.

## Important — submission preparation and reviewer concerns

- Sensors remains the working target from the supplied manuscript template. Its novelty and scope assessment remains editorial. [Official instructions](https://www.mdpi.com/journal/Sensors/instructions) and [ethics policy](https://www.mdpi.com/ethics) were checked through indexed official text; direct retrieval failed. This does not certify every submission-portal requirement.
- Upload the verified supplement with the submission or deposit it in an author-approved accessible archive. GitHub public accessibility was not established; an unauthenticated API check returned 403. Do not claim an accessible public archive until verified.
- The retained title needs its explicit scope qualifications. Primary adolescent evaluation uses retraining. The additional frozen test uses newly trained glucose-only models on AZT1D. Warm replay does not validate live sensing, an emergency system or clinical utility. A reviewer may still request a narrower title.
- Preserve negative findings: no demonstrated HR benefit, no universal stack superiority, and no demonstrated LLM advantage over deterministic rendering. Small convenience cohorts, overlapping training folds, complete-case selection, timestamp limitations and restricted tuning constrain generalization.
- The synthetic detector audit was designed and inspected in the same implementation process, not by a blinded external evaluator. Do not relabel it as independent clinical adjudication.

## Optional — extensions beyond the current claims

Broader context lengths and foundation-model tuning, additional independent multimodal cohorts, independent detector evaluation, and usability studies could strengthen future work. Prospective clinical studies become relevant if claiming patient benefit or treatment safety; those claims are excluded from this paper and are outside the available project scope.

No acceptance guarantee or immunity from reviewer criticism is claimed. The remaining critical items are author decisions and declarations, not an instruction to manufacture positive experiments.
