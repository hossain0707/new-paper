# Supplementary File S1

Current study: Qwen-primary generation audit plus availability-aware glucose forecasting. Historical DeepSeek structured responses were not recovered; older narrative CSVs are not substitutes. No exact DeepSeek reproduction or cross-model numerical comparison is claimed.

Start with SOURCE_TERMS.md, REPRODUCE.md, audit/VERIFIED.json and references/reference_verification.json. Qwen/ contains all 999 test decision/timing rows (888 LLM plus 111 deterministic) and 216 development rows, with pseudonymous participant groups. These rows permit independent aggregate and participant-bootstrap checks without exposing copied measurements. Stack/ contains 690 fit manifests, hyperparameter searches/training histories and 138 validation-weight files. audit/ records read-only split, weight and checkpoint verification. derived/ and project/results/ contain reported cohort and saved participant summaries; historical neural HR results are summary-based.

The scientific scripts are the actual executed code. Source terms and source hashes govern reproduction. Saved summaries are not mislabeled as raw predictions. Model manifests distinguish checkpoint identities from served aliases. Figure 2 was regenerated from Qwen and replayed synthetic contracts; Figure 5 was regenerated from recovered RF raw predictions; the other figures derive from retained forecasting summaries.

Public material excludes participant-bearing payloads and raw time series. Their SHA-256 manifests are included, and regeneration commands are provided. A fresh model run can differ numerically despite matched settings; it must be saved separately, never overwrite evidence.
