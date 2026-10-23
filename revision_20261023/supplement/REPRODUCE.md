# Reproducing the Qwen primary revision

Work in a new checkout or copy, never the retained evidence directories. Commands below assume Linux, Python 3.12, a working NVIDIA/PyTorch environment and `cd supplement/project`. Read SOURCE_TERMS.md before acquiring data. No dataset or model weights are included. Public derived evidence carries source attribution and applicable noncommercial restrictions.

The executed Spark environment used GB10, driver 580.95.05, CUDA 13.0, PyTorch 2.13.0+cu130, numpy 1.26.4, pandas 2.2.3, scipy 1.15.2, scikit-learn 1.6.1, xgboost 2.1.3, shap 0.47.2 and pyarrow 19.0.1. `requirements-revision.txt` records the historical forecasting environment separately; its Torch 2.7 H200 environment is not interchangeable with Spark. The neural Stack sensitivity changes execution environment as well as validation size. Preserve image/weight manifests; package versions alone do not reproduce a container. Existing loaded Qwen service was never stopped or replaced.

Acquire and prepare pinned sources:

```sh
python3 scripts/acquire_ddatshr_revision.py
python3 scripts/acquire_revision_data.py
python3 scripts/prepare_revision_data.py --dataset all --history 120
python3 scripts/acquire_revision_azt1d.py
python3 scripts/prepare_revision_azt1d.py
```

Required prepared array SHA-256: DDATSHR `cbe727f2edf45ab0145774e03f235f34d2fa7c13c82d49c340c98a3d0458c7d9`; DiaData `54a12bfb016944a999ef51be91f749ba48c51b95c895b3a3d6bea1f893a4a35d`. Stop on mismatch. Original acquisition scripts save checksums and provider metadata. Primary origins must remain 72,414/14,500 from 17/29 people.

The historical primary fit code is included (`run_revision_forecasts.py`, `src/revision/experiment.py`, `reporting.py`). A new full primary run must use a separate `--output results/revision_v4/new_primary_reproduction`, with `--gpus 0 --workers-per-gpu 1` on a single GPU. Do not run its four-GPU defaults on Spark. Saved primary tables are historical aggregate evidence, not a newly rerun complete primary neural experiment.

Rebuild RF and language cases using the executed recovery path:

```sh
python3 scripts/recover_third_review_rf.py --seeds 42 --workers 12 --output results/revision_v4/recovered_rf_seed42
python3 scripts/build_revision_cases.py --fit-root results/revision_v4/recovered_rf_seed42 --output results/revision_v4/recovered_cases
```

This deterministic sampler rebuilds the case-selection CSV from verified prepared records. Its required SHA-256 is `bf916d2061e2a3a114da665d409dfd8e569b1a25e95dcd3a095a4d940fd16f3e`. It must yield 135 cases, 24 development cases from eight people and 111 test cases from 38 other people. Copy `../protocols/historical_prompt_freeze.json` to the case folder after removing the added `historical_model_field_scope` annotation; its examples are the fixed case IDs `115db9d75d6a8000`, `d92ef00d7fa6f0d2`, `159b450c59087c0b`. This historical freeze preserves selection/prompt identity; it does not make Qwen a DeepSeek reproduction. Model identity for current requests comes from `../models/Qwen/upstream_verification.json` and the Qwen configurations.

Serve only the exact pinned NVIDIA repository revision `b9ce18a6887e18c089fdea8509a33e63429a3300`. Verify all 21 weights and five metadata/tokenizer files against the included manifest. The served alias is `nvidia/Qwen3.8-Flash-Next-FP4-Local`; internal architecture is Qwen4Exp. vLLM `0.1.dev21830+g8244d1c44.d20261015`, TP1, context 262144, modelopt_mixed, bfloat16, FP8 KV, prefix caching, four-token MTP. Configured structured backend auto resolves XGrammar 0.2.8 through the installed validation path. Re-run the nonparticipant conflict probe before accepting schema enforcement. There is no hosted inference API substitution.

Run generation after forecasting stops. Keep every response/failure and use a new directory if any setting differs. An optional API key belongs in a local secret file passed through `--api-key-file`; never commit it. Use loopback only.

```sh
python3 scripts/run_review_comparison.py --model nvidia/Qwen3.8-Flash-Next-FP4-Local --checkpoint-id local_weight_manifest_sha256:3ab1017efc86560384dcc3ed3a60010d442f29d678144ea24d5ddc9ee715090d --case-source results/revision_v4/recovered_cases --split development --output results/revision_v4/new_qwen/development
```

Inspect development integration and freeze configuration, response hashes, protocol and choice of unchanged prompts before test. Do not tune using test. Then use the same command with `--split test --output results/revision_v4/new_qwen/test`. A new run is a reproduction attempt, not replacement of recorded outputs; inference engine/hardware nondeterminism can matter.

```sh
python3 scripts/summarize_third_review_qwen.py --input results/revision_v4/new_qwen/test --case-source results/revision_v4/recovered_cases --output results/revision_v4/new_qwen/test_summary
```

The frozen validator (`src/revision/grounding.py`) SHA-256 is `78019ae3a78a70352325f85c0756d9d036d189c30bd08a564913d7cfd4fa93cc`. Failures count against LLM acceptance despite deterministic fallback. The public `Qwen/*_request_audit.csv` enables summary reanalysis from recorded decisions/timings without releasing measurement-bearing payloads. `scripts/independent_audit.py` independently revalidates full private raw records, prompt/schema identity, participant separation, all fit splits and validation-only weights once those artifacts exist. It requires the documented directory layout and output path that does not yet exist. Its representative neural checkpoint verification uses the original GPU and original batch boundaries; the initial CPU replay differed slightly and was not passed off as exact.

Run full three-validator Stack separately from generation:

```sh
python3 scripts/run_third_review_stack.py --tree-workers 12 --gpu-workers 2 --output results/revision_v4/stack_full
python3 scripts/summarize_third_review_stack.py --input results/revision_v4/stack_full --output results/revision_v4/new_stack_summary
```

Expected: 690 base fits, 138 validation-only weight sets, 414 horizon-specific weight optimizations, no overlap of training/validation/test people. Weights use equal-participant validation squared error on a simplex. Checkpoints, validation/test predictions, searches and training histories are saved. Do not infer full neural HR raw reproduction from this CGM+HR-only sensitivity.

For RF clinical/coverage verification, recover all three seeds into `results/revision_v4/recovered_rf_primary`, then run `scripts/verify_third_review_rf_raw.py`. It refuses an existing verification output and compares every participant/horizon against saved MAE, interval and TP/FP/FN summaries. Historical neural HR extension uses saved participant summaries in `scripts/analyze_third_review_saved.py`; label it accordingly. AZT1D transfer code (`run_revision_strengthening.py --mode external`, `freeze_revision_external.py`, `evaluate_revision_external.py`) retains its historical four-GPU assumptions; inspect and configure a distinct reproduction environment before running, never silently relabel hardware.

Figures: `plot_third_review_saved.py` generates the retained aggregate forecast figures; its superseded DeepSeek Figure 2 must not be used. Run `../scripts/plot_qwen_primary.py PROJECT_ROOT NEW_FIGURE2_FOLDER` to regenerate Qwen Figure 2 and synthetic replay, and `../scripts/plot_recovered_rf.py PROJECT_ROOT NEW_FIGURE5_FOLDER` for the recovered raw-RF Clarke figure. These are the current Figure 2/5 scripts. All current PNG/PDF/SVG assets are in `figures/` with source hashes. Syntax and software tests are distinct from empirical reproduction.

Missing evidence: original structured DeepSeek cases/responses, raw historical neural HR predictions, independent human usefulness evaluation, prior-publication editorial assessment, cold-start/deployment cost. They are not fabricated by these commands. Author contributions, conflicts, consent/institutional wording and independent human verification remain author decisions. Funding is user-confirmed as no external funding.

Public Qwen summary replay: `python3 scripts/reanalyze_public_qwen.py SUPPLEMENT_ROOT NEW_OUTPUT_FOLDER`. The published pseudonymous participant labels retain original bootstrap order. All 888 request decisions and all three factor intervals/p values were reproduced from this public CSV. `audit_fit_identities.py PROJECT_ROOT` additionally verifies all 690 source/protocol/metadata hashes and winning validation configurations.
