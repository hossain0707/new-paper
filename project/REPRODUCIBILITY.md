# Reproducing the corrected research

Run commands from `project/`. Revision-v2 artifacts are separate from historical
`results/tables/`, old manuscript drafts and smoke tests. The primary source is
`manuscript/manuscript_REVISED_EVIDENCE.md` once generated, with its matching Word file
at the repository root. The historical LLM runner is disabled because its inputs
contained null SHAP and invented intervals.

## Environment

The forecasts used Python 3.12 and the NVIDIA PyTorch 25.03 container, with four H200
GPUs. Exact package versions are saved in
`results/revision_v2/provenance/forecast_environment.json`. The special NVIDIA Torch
build is supplied by that container; the pip requirements intentionally do not pretend
that its version is an ordinary downloadable PyPI wheel. Other PyTorch builds are
portability targets, not bitwise-equivalent environments.

```bash
python -m venv --system-site-packages .venv-revision
.venv-revision/bin/python -m pip install -r requirements-revision.txt
source .venv-revision/bin/activate
python -m unittest discover -s tests -v
```

The session's system packages already matched the pinned versions; only SHAP and its
slicer dependency needed installation in the isolated revision environment. GPU kernels
use deterministic algorithms and math attention. Cross-platform equality is not asserted.

## Data acquisition and preparation

```bash
python scripts/acquire_ddatshr_revision.py
python scripts/acquire_revision_data.py
python scripts/prepare_revision_data.py --dataset all --history 120
python scripts/prepare_revision_data.py --dataset all --history 240
```

Acquisition verifies DDATSHR Git/Git-LFS hashes and the DiaData archive's published MD5.
The exact DiaData raw file is `SDBII_15min_sampling_raw.csv`; the similarly named
preprocessed file must not be substituted. No raw records are bundled with this code.
Observe each source's licence and terms. Internet access is required to acquire the
public sources and model weights, but generation requests remain local.

Prepared arrays, sample identifiers, retention tables and clock/parser audits are saved
under `data/processed/revision_v2/history_*/`. The primary input length is nine boundaries,
with four channels; CGM-only removes HR on the same records. DiaData inputs have an extra
15-minute availability delay. Forecast horizons refer to the documented bin labels.

## Forecasts and exploratory sensitivity

```bash
python scripts/run_revision_forecasts.py --datasets ddatshr --workers-per-gpu 4
python scripts/run_revision_forecasts.py --datasets diadata --workers-per-gpu 2
python scripts/run_revision_sensitivity.py --mode source
python scripts/run_revision_sensitivity.py --mode history
python scripts/run_revision_sensitivity.py --mode history_short_matched
```

The first two commands were run concurrently in this revision. They fit 612 and 1044
primary patient/model/seed/condition tasks. Each learned model is evaluated at the same
three horizons jointly. Complete manifests are written only after checkpoints, validation
predictions and test predictions are saved. Rerunning identical tasks reuses completed
artifacts. Different code/data/epoch configurations require a fresh output directory;
the runner rejects stale completion manifests. Interruptions may leave an incomplete
folder, which the next invocation retrains.

The three-seed primary stack is constructed only after all base fits finish. Its five
members exclude Ridge. Source transfer withholds all patients of one source from fitting
and validation. History sensitivity refits both lengths on identical longer-history
eligible sample manifests. These results must not be substituted for primary results.

## Local generation

The separate existing vLLM environment contains vLLM 0.27.1 and its own dependencies.
Do not mix it with the forecasting environment. In this workspace an inherited
`PYTHONPATH` points to unrelated user packages; clear it for this service.

From the workspace root, after obtaining the stated model revision:

```bash
PYTHONPATH='' PYTHONNOUSERSITE=1 llm_server/vllm_env/bin/vllm serve \
  /absolute/path/to/DeepSeek-V4-Flash-0731/revision-7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 --port 8002 --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.55 --kv-cache-dtype fp8 \
  --max-model-len 8192 --enforce-eager --max-num-seqs 8
```

From `project/`, once the server is healthy and the RF case checkpoints exist:

```bash
python scripts/build_revision_cases.py
python scripts/audit_revision_guardrail.py
python scripts/run_revision_llm.py --split development
python scripts/run_revision_llm.py --split test
python scripts/summarize_revision_llm.py
```

The selection is fixed independently of error. `--available` allows a partial pass over
already-built, preselected cases while remaining RF fits run; final reporting still
requires every selected test case. Few-shot examples come from the first selected
development participant in dataset/identifier order, one for each available state.
The prompt, detector hash, selection hash, model revision and decoding settings are
frozen and checked on subsequent runs. Do not tune them after inspecting final test
outputs and then present those same cases as untouched test data.

The response archive preserves full requests, model responses, request errors, rejection
reasons and fallback origin. An error response is not silently retried or erased. Cases,
responses, arrays and checkpoints are retained locally rather than placed in Git. The
synthetic detector challenge contains no patient records and is included in the evidence
archive. Its boundary tests deliberately demonstrate that source corruption is outside
the gate's verification capability.

## Results, manuscript and figures

```bash
python scripts/summarize_revision.py
python scripts/analyze_revision_errors.py
python scripts/plot_revision.py
python scripts/write_revision_manuscript.py
python scripts/build_revision_docx.py
python scripts/write_revision_response.py
python scripts/verify_revision_artifacts.py
python scripts/package_revision.py
```

The summarizer refuses incomplete fits. Final numerical prose and manuscript tables are
derived from the same CSVs used by the figures. Figures have PDF, SVG and 600-dpi PNG
versions. The private professor-response folder is excluded from the public supplement.
Clinical review, author approval, institutional determinations and journal submission
are not simulated by these commands.

## Expected limitations

One inner participant, convenience samples, overlapping training folds, complete-case
selection and upstream DiaData time rounding constrain inference. Generation is a
controlled schema-copying experiment with a deterministic renderer, not validated clinical
advice or a demonstration of LLM utility. The original six-modal study's claims are not
verified by rerunning this reduced-input pipeline. See the manuscript and literature audit.

## Additional v3 evidence (separate from primary v2)

`protocol_v3.md` was written after the v2 findings were known, before the new experiments. Do not edit frozen protocol/training code in place and reuse cached fits. The 18 development fits and 276 sensitivity fits have separate outputs. The original 1,656 fits and 18 primary tests remain unchanged.

1. Run `.venv-revision/bin/python scripts/acquire_revision_azt1d.py` to acquire and verify AZT1D v1 from https://data.mendeley.com/datasets/gk9m674wcx/1 (CC BY 4.0). Save the provider ZIP as `data/raw/azt1d_v1/AZT1D_2025.zip`; verify SHA-256 `2c6b4183fdc729c708c1b3230ab1988d2c9006dd35729537f2de07f649f0d113`. The public file API and download URL are in `results/revision_v3/provenance/azt1d_provider_files.json`. Source records are not redistributed. Subject 14 lacks a CGM-specific field and is excluded before scoring; see the schema decision.
2. Run `.venv-revision/bin/python scripts/run_revision_strengthening.py --mode external --workers 6`, then `scripts/freeze_revision_external.py` with the same interpreter. The freeze script refuses to overwrite an existing freeze. For a fresh reproduction, preserve the supplied freeze as a reference outside the output directory before creating a new run. New timestamps and script environment may change byte hashes; compare scientific outputs separately.
3. Run `scripts/prepare_revision_azt1d.py`, then `scripts/evaluate_revision_external.py`. The evaluator verifies every frozen artifact before scoring; it does not tune on the external outcomes and refuses to overwrite completed predictions.
4. Run `scripts/run_revision_strengthening.py --mode validation --workers 10` for all 276 three-validator RF fits.
5. Create `.venv-chronos` with `python -m venv --system-site-packages .venv-chronos` in the same NVIDIA base container and install `requirements-chronos.txt`. Run `.venv-chronos/bin/python scripts/run_revision_chronos.py`. Model `amazon/chronos-2` revision `29ec3766d36d6f73f0696f85560a422f50e8498c` is pinned before inference. No cross-learning, task fitting, future covariates or external adaptation is used. All original primary origins are retained. Pretraining overlap is unknown.
6. Run `.venv-revision/bin/python scripts/summarize_revision_strengthening.py` and `scripts/plot_revision_strengthening.py`. New bootstrap intervals and comparisons are descriptive; no new p-values enter the primary multiplicity family.
7. Start the original frozen loopback LLM server using the documented TP4 configuration. After all GPU forecasting finishes, run `.venv-revision/bin/python scripts/measure_revision_latency.py`. This performs 90 sequential measured executions plus three excluded warmups. Per-case models and histories load before timing. The scope is available-history-to-rendered-output warm processing, not device acquisition or prospective clinical validation. Keep all outputs, failures and fallback times. Existing response files are preserved for interrupted-run resumption; a fresh independent timing run must use a separate output location rather than mixing sessions.
8. After replay finishes, run `scripts/verify_revision_strengthening.py --gpu`, then `scripts/verify_revision_strengthening.py --final`. Regenerate manuscript/Word/PDF, run the primary artifact verifier with `--final-only`, and rebuild the supplement. All 18 external checkpoints are reloaded for a 2,048-example prediction check. GPU batch size/backend is fixed to match original numerical execution.

The additional figures and tables live under `results/revision_v3`; the clean paper retains its original title and section organization. No clinical trial, new participant recruitment or clinician adjudication is claimed. Author approval, truthful declarations, institutional secondary-use determination and prior-publication disclosure still require the responsible authors.
