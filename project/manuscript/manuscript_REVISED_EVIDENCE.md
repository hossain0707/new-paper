# Multimodal Deep Learning for Real-Time Glucose Prediction with Zero/Few-Shot On-Premises LLM Explainability: External Validation on an Independent Adolescent Cohort

**Md. Najmul Hossain ¹, Jong-Nam Kim ¹,***

¹ Department of AI Convergence, Pukyong National University, Busan 48513, Republic of Korea

* Correspondence: kjnam@pukyong.ac.kr

## Abstract

GlucoSense AI combines glucose forecasting with on-premises LLM explainability. We evaluate its forecasting approach on an independent adolescent cohort and assess explanation without task-specific LLM fine-tuning. This retrospective study evaluates reduced-input models, with additional frozen-model transfer and processing replay. Matched observed-target evaluation retained 17 DDATSHR adolescents (72,414 origins) and 29 non-DDATSHR DiaData participants (14,500 origins). Random Forest (RF), other baselines, three neural variants and constrained ensembles used patient-disjoint inner validation and three training seeds. RF subject-macro mean absolute errors at 30/60/90 minutes were 18.12/30.54/38.82 mg/dL on DDATSHR. Stacking improved RF at all three DDATSHR horizons after Holm correction, but not significantly on DiaData; adding HR showed no significant benefit. Genuine held-out RF TreeSHAP supplied 111 patient-separated test cases to a local LLM. Complete-contract acceptance was 100/111 for zero-shot and 96/111 for few-shot prompting, versus 111/111 for deterministic rendering. A synthetic challenge rejected 300/300 contract violations but accepted 30/30 corrupted source inputs. Additional frozen-model evaluation used 24 AZT1D participants. These findings quantify forecasting and source-relative consistency without establishing clinical safety or live deployment performance.

**Keywords:** glucose prediction; continuous glucose monitoring; multimodal deep learning; large language models; zero-shot prompting; explainable AI; external cohort evaluation

## 1. Introduction

Continuous glucose monitoring provides a basis for forecasting future glucose changes, but a prediction alone does not explain its inputs or limitations. GlucoSense AI was proposed to combine physiological signals, ensemble forecasting and locally generated explanations. The present paper continues that research goal: evaluate the forecasting approach under different recording conditions and assess whether its explanation component can operate through zero/few-shot prompting without task-specific parameter updates.

The independent adolescent DDATSHR cohort [1] is the primary focus. Its observed CGM and heart-rate signals support a reduced two-modality implementation, not reconstruction of every modality in the earlier system. A second analysis uses the non-DDATSHR portion of DiaData. Forecasting, attribution and language-output checks are evaluated separately so a faithful description of an inaccurate forecast is not counted as clinical validation.

Scope of the title: “external validation” describes evaluation of the GlucoSense forecasting approach on an external cohort with explicitly retrained models and patient-held-out testing. It is not external validation of a frozen original prediction model. “Real-time” identifies the framework's intended use; the present experiments are retrospective and do not validate live sensing-to-alert deadlines. A separately labelled secondary experiment freezes newly trained DDATSHR glucose-only models before testing AZT1D; a processing replay measures available-history-to-output latency. Neither retroactively validates the unavailable historical model or live sensing. These qualifications apply throughout the manuscript.

### 1.1 Relationship to prior work and new contributions

This study follows an earlier GlucoSense publication [2], but constitutes newly trained, reduced-input retrospective experiments. The earlier six-modal checkpoints, modality reconstruction outputs and original evaluation predictions are unavailable in this repository. We therefore do not treat its reported OhioT1DM accuracy as independently verified evidence, do not reuse its architecture artwork, and do not interpret a numerical difference from that article as a causal modality effect. The supplied OhioT1DM documentation also does not substantiate the raw wrist-PPG reconstruction pathway described previously [3]. These provenance limitations require a different scope from a locked-model external validation claim.

The new evidence addresses four components of the original paper: (1) matched adolescent-cohort forecasting with all three reduced-input deep models and the full constrained stack; (2) CGM-only, history and source-transfer comparisons; (3) actual zero/few-shot local generation from held-out RF predictions and additive TreeSHAP, with a deterministic comparator; and (4) patient heterogeneity and explicit filter false-negative boundaries. These are empirical contributions using established components. The study does not claim a new neural architecture or first use of glucose-related LLM explanations.

### 1.2 Motivation

Two questions follow directly from the original study's stated limitations: how does the forecasting approach perform in a cohort collected under different conditions, and can its explanation module run without task-specific fine-tuning, with what reliability? This paper addresses both empirically. H1 tests whether observed HR improves RF over CGM-only inputs on matched windows. H2 tests stack versus RF and RF versus persistence. H3 examines detectable contract errors and the acceptance coverage of zero/few-shot generation. A favorable answer is not assumed; negative and uncertain results are retained.

## 2. Related Work

### 2.1 CGM-based glucose prediction

Kolev et al. present a broad multi-method glucose benchmark [4]. Their Methods specify that the reported experiments used physics-based synthetic datasets because original downloads were unavailable. This differs from the observed DDATSHR and DiaData records examined here. Published RMSE values from that work are consequently not inserted into our matched comparison. More generally, model rankings across papers cannot control for patient identity, target construction, history length or data access.

The neural models draw on established recurrent, convolutional and attention components. Temporal Fusion Transformers (TFT) were developed for general multi-horizon forecasting [5]; their original paper is an architectural reference, not evidence of a glucose-specific improvement. Our TFT-inspired encoder omits parts of the original model and is labelled accordingly. XGBoost and tree attribution follow their original method descriptions [6, 7]. No performance improvement attributed to HRV in unrelated prior work is assumed.

Recent GlucoFM-Bench work benchmarks foundation models across multiple glucose datasets [8]. Its preprint results motivate a modern comparator but are not directly comparable to this study’s observed-only patient-held-out protocol. We therefore evaluate pinned Chronos-2 [9] on exactly our primary test origins, and do not claim the first multi-dataset or foundation-model glucose benchmark.

### 2.2 Multimodal physiological approaches

Cinar and Maleshkova introduced DiaData and subsequently studied quality-enhanced hypoglycemia classification [10, 11]. Their processing includes outlier handling and gap interpolation. Our regression task instead uses the raw export and observed targets, retains source labels and evaluates generalization to held-out people. It is not a replication of their classifier or a claim that interpolation is inappropriate for every retrospective application. HR refinement remains an active topic, including a recent preprint on short-term hypoglycemia prediction [12].

### 2.3 Explainability, LLMs, and emergency alerts

Alredaini et al. compare conventional, deep and language-model glucose forecasting for type 2 diabetes, including SHAP and language explanations [13]. GlucoLens examines postprandial prediction and counterfactual support using wearable and dietary information [14]. These studies rule out a broad first-use claim for language models or explanation in glucose research. Our narrower emphasis is a machine-auditable contract, patient-separated generation cases, a deterministic comparator and a measured source-truth failure boundary.

TreeSHAP attributes a model prediction to its inputs [7]; it does not establish physiological causation or justify emergency escalation. Clinical alert effectiveness requires evidence about events, actions and outcomes beyond a retrospective error grid or a source-consistency check.

### 2.4 Zero/few-shot clinical LLM deployment

The local zero/few-shot configuration studied here avoids task-specific LLM gradient updates. That implementation fact does not demonstrate equivalence to a fine-tuned system, reduced total deployment cost or clinically useful explanations. Existing glucose/SHAP/LLM work [13, 14] also prevents a broad novelty claim. We therefore compare the optional LLM step directly with deterministic rendering and disclose its contract failures.

## 3. System Architecture

The paper retains the GlucoSense sequence of data acquisition, preprocessing, forecasting, explanation and presentation. Its evaluated implementation is restricted to components supported by the available records and executable experiments. Six-stream wearable acquisition, EHR extraction, a production application and emergency dispatch are not presented as tested components of this revision.

### 3.1 Five-layer architecture

The original GlucoSense design [2] has five layers: Data Sources, Preprocessing, Model Inference, Risk Scoring and Emergency Alerts, and Application. This organization is retained as the framework. In the evaluated implementation, Data Sources supplies recorded CGM and HR; Preprocessing constructs observed histories and targets; Model Inference fits the forecasting models and supplies RF attribution to the local LLM. The fourth layer is evaluated only through a study-score consistency task, without emergency dispatch. The Application layer remains a deployment objective; dashboards, portals and live clinical services are not evaluated here. Figure 1 shows the executed research workflow within this framework. The forecasting ensemble and the explained RF are distinguished explicitly; the language model does not explain the stack.

### 3.2 Multi-modal data processing

The original design describes CGM, HRV/vitals, sleep and EHR processing streams [2]. The adolescent records do not contain all of these streams. Consequently, this evaluation implements observed CGM/HR processing and tests its reduced models; it does not claim that imputed SpO₂, blood pressure, sleep or EHR features reproduce the historical 88-dimensional representation.

Each primary input contains nine boundaries spanning two hours. Channels are CGM, HR and the sine/cosine of time of day. For participant p, forecast origin t and history index k, X(p,t,k,c) denotes channel c at the k-th available historical boundary. Targets Y(p,t,h) are observed glucose values in the bin labelled t+h, where h is 30, 60 or 90 minutes. This notation consistently distinguishes participant, origin, history and channel indices. The actual target reading can precede the bin boundary; horizons refer to the stated grid convention.

All historical channels and all three future glucose targets must be finite. A participant needs at least 50 eligible origins. The same patient/timestamp manifest is used by every primary model and by the CGM-only ablation, which removes HR from the input without changing eligibility. This isolates its conditional predictive contribution among HR-observed windows; it does not estimate benefit in participants whose HR is absent. Complete-case selection can favor better-recorded periods and is reported explicitly.

For the primary two-modality analysis, the input dimensions are explicitly:

  X(p,t) = [X(p,t,k,c)] with k = 0,…,8 and c ∈ {CGM, HR, sin(time), cos(time)}.     (1)

### 3.3 Emergency alert pipeline and composite risk score

The earlier architecture included multi-channel emergency alerts and a score involving unavailable physiological channels. That original composite score cannot be validated here, and no push, SMS, email or phone alert was dispatched in these experiments. To test categorical consistency without inventing those inputs, the revised explanation experiment uses a declared artificial study score and numbered tiers, not clinical “safe” or “emergency” labels.

The deterministic, zero-shot and few-shot conditions return the same declared JSON contract: latest available glucose/HR and input delay, three forecasts, recent rate, a study-only score, three SHAP entries, a tier and a direction. The score is an explicitly artificial consistency task: its base is 85 at glucose ≤70 or ≥250, 55 at ≤80 or ≥200 otherwise, and 20 elsewhere; min(15,5×absolute recent rate) is added and the result capped at 100. Half-open boundaries 40/60/80/95 map it to five numbered tiers. Direction compares the 90-minute forecast with the latest input using ±5 mg/dL. Neither the score nor its tiers constitute a validated clinical triage protocol.

## 4. Datasets

DDATSHR remains the primary independent adolescent cohort in the paper. DiaData is a secondary source-separated analysis, not another adolescent cohort and not assumed independent of every source mentioned in the earlier publication.

### 4.1 DDATSHR — Diabetes Adolescents Time Series with Heart Rate

DDATSHR was acquired at Git commit 36379d9397d05bb00ac20cf91a5a2b648fecfafe. The Abbott, Medtronic and population CSV files match upstream Git objects, and the HR file matches its Git-LFS SHA-256 object. The parser quarantines records whose field count disagrees with the header; it does not silently shift columns to accommodate unquoted text. Source hashes and quarantined line numbers are retained. Glucose in mmol/L is converted using 18.0182 mg/dL per mmol/L.

For DDATSHR, bins are right-closed and right-labelled at 15-minute boundaries. The latest observed CGM within each completed bin and mean observed HR are used; no interpolation or forward filling is performed. The 15-minute grid respects the slower Abbott stream rather than requiring an observation every five minutes. Offset-free timestamps are excluded. The ambiguous 27 October 2019 interval is excluded, along with Medtronic records until the first subsequent recorded clock adjustment plus two hours, and two-hour neighborhoods of other clock adjustments. All Medtronic participants with post-transition glucose have a recorded subsequent adjustment. Local-date CGM and UTC-date HR exclusions are conservatively applied as specified in code; remaining device-clock uncertainty cannot be eliminated retrospectively.

### 4.2 DiaData — Route to Larger-Scale Validation

DiaData uses Zenodo record 17285631 and the raw SDBII file within the 15-minute archive. The downloaded archive matches its published MD5 checksum, 2c876bf0b8bc9ad6c7823e672287b216. All DDATSHR-source participants are removed before analysis. D1NAMO and HUPA-UCM remain distinguishable. Because the earlier GlucoSense article cited D1NAMO as a development source, DiaData is not described as universally independent of that historical development process. The models evaluated here are newly fitted and do not use previous GlucoSense weights.

DiaData is a raw integrated export, not untouched device output. Its upstream integration code rounds timestamps, and original reading times are unavailable in the 15-minute file. We delay its entire input sequence by one grid step, so the latest supplied input label is t−15 minutes and targets remain labelled t+30/t+60/t+90. This is an availability safeguard under a within-bin aggregation assumption, not proof of exact source-clock accuracy. DDATSHR needs no extra delay because original offset-bearing records are available. Cross-cohort absolute errors therefore describe distinct populations and availability conventions, not a controlled experiment on age or sensor hardware.

### 4.3 Cohort retention and matched eligibility

The corrected primary analysis retained 17 DDATSHR participants (72,414 origins) and 29 DiaData participants (14,500 origins). The Medtronic parser quarantined 1,540 malformed records. DiaData excluded all 18 DDATSHR-source participants before eligibility assessment. Excluded participants and per-source availability are retained in the supplementary tables.

**Table 1.** Source-specific eligibility after observed-input/target and minimum-sample requirements. Primary cohorts require CGM and HR; additional AZT1D evaluation uses glucose only.

| Cohort | Source | Source participants | Retained | Eligible origins |
| --- | --- | --- | --- | --- |
| ddatshr | DDATSHR | 17 | 17 | 72,414 |
| diadata | D1NAMO | 9 | 7 | 681 |
| diadata | HUPA-UCM | 24 | 22 | 13,819 |
| azt1d (additional) | AZT1D | 25 | 24 | 92,976 |

The primary DDATSHR cohort includes both Abbott and Medtronic data. Participant 907 has mixed actual sensor records despite the population file's single-brand label. The corrected retention therefore does not support interpreting ordinary Abbott intervals as pathological fragmentation. The quarantine and clock exclusions precede model fitting and are independent of observed model errors.

For an additional frozen-model test, AZT1D version 1 [15] provides 25 individuals with type 1 diabetes using automated insulin delivery. Its CC BY 4.0 archive was checksum-verified. Twenty-four CSVs contain a CGM-specific field; Subject 14 contains an ambiguous CGM/BGM field and was excluded before scoring. Using the same nine-boundary observed-only glucose history and all three observed targets retained 24 participants and 92,976 origins. Provider timestamps are local Arizona. No HR, insulin, meal or calendar covariates enter this additional test; it is not described as an adolescent-only cohort. Source processing aligns pump events, but the present analysis uses the documented CGM field only. Full retention and schema decisions are supplied.

## 5. Model Architectures

### 5.1 Deep learning base models

The TFT-inspired encoder has feature selection, a 64-unit LSTM, four-head attention, gated residual blocks and 0.1/0.5/0.9 quantile heads. Its median supplies the point forecast. BiLSTM–Transformer uses two 64-unit bidirectional LSTM layers and two Transformer encoder layers, four attention heads and feed-forward width 256. CNN–GRU uses modality-specific convolutions of widths 32/64 followed by a two-layer 64-unit GRU. The CGM-only CNN variant has a single branch and consequently differs in parameter count; its ablation is not a strictly capacity-matched modality intervention. None is claimed to reproduce an unavailable historical six-modal checkpoint.

### 5.2 XGBoost, Random Forest, and stacked ensemble

The persistence baseline repeats the latest available glucose. Linear trend extrapolates the last two available CGM values without clipping; its lead accounts for the extra DiaData input delay. Ridge, RF and XGBoost use the flattened history. Ridge alpha is selected from 1, 10 and 100. RF uses 300 trees, a minimum leaf size of five and candidate depths eight and sixteen. XGBoost uses 300 estimators, learning rate 0.05, row and feature subsampling 0.8, and candidate depths three and six. Validation MAE averaged across horizons selects the configuration.

Seeds 42, 1729 and 2026 are retained individually and combined by averaging predictions for the primary comparison. The five-member stack comprises RF, XGBoost and the three neural models. For each horizon and seed, weights solve validation squared error subject to w_j ≥ 0 and Σ_j w_j = 1. The optimization is directly constrained; no unconstrained coefficient renormalization is used. A two-member RF/XGBoost stack is a secondary ablation. Ridge remains a standalone baseline. Inner predictions, selected configurations, weights, training histories and checkpoints are archived.

For participant p, origin t and horizon h, the ensemble is:

  ŷ(p,t,h) = Σ_j w(p,h,j) f_j(X(p,t),h), with w(p,h,j) ≥ 0 and Σ_j w(p,h,j) = 1.     (2)

Here j indexes base models, f_j(X(p,t),h) is model j’s absolute-glucose forecast at horizon h, k in Section 3.2 indexes historical boundaries, and c indexes channels. The training-seed index is suppressed for readability. Weights are fitted separately for each outer fold, horizon and seed using only its inner participant, not fixed historical coefficients.

### 5.3 SHAP explainability

The narrative experiment explains the held-out RF at seed 42, not the stack. TreeSHAP values are computed from the saved RF checkpoint on each selected input. The sum of all attributions plus the expected glucose-change prediction is checked against the RF output. The three largest absolute contributions at 30 minutes are supplied with feature identities, lags and mg/dL contribution units. They describe the model's change prediction and do not identify causal physiological drivers [7].

The checked additive decomposition for the explained RF is:

  ŷ_RF(p,t,30) = G_last(p,t) + E_p[f_delta(X)] + Σ_i φ_i(p,t,30).     (3)

Here i identifies a flattened history-channel feature, φ_i is its TreeSHAP contribution to the predicted glucose change, and G_last is the latest available CGM input. The expectation is the TreeSHAP reference output of the RF fitted without participant p. This feature index differs from ensemble-member index j and historical-boundary index k. The case archives retain all contributions, even though generation receives only the top three.

### 5.4 Training convergence

Neural inputs and glucose-change targets are standardized using base-training data only. The model predicts change from the latest available glucose, which is added back for evaluation. Learning rates 0.001 and 0.0003 are tested with Adam, batches of 1024, gradient clipping at one, at most 80 epochs and patience ten. The checkpoint with lowest validation MAE is restored. TFT uses quantile pinball loss; the other neural models use mean squared error. Loss and tree-fit weights are inversely proportional to each training participant's window count, normalized to mean one. This balances subjects during fitting. Differences therefore reflect model-and-loss configurations, not an isolated architecture intervention.

Selected epochs, learning rates and validation histories are retained per fit. Early stopping and checkpoint restoration are reported as implemented; they do not prove that every architecture converged globally or that overfitting is absent. Test patients were not used to select stopping epochs.

### 5.5 Reduced Feature Configuration for DDATSHR

The same nine-boundary CGM/HR/time representation is used for trees and deep models in each cohort. The CGM-only comparison removes HR while retaining the identical eligible origins and targets. Both cohorts include all three neural variants; deep models are no longer evaluated on a smaller window panel than the trees. The unavailable six-modal representation is not reconstructed or treated as validated by this reduced implementation.

Additional protocol v3 was recorded after all primary outcomes were known, before its own evaluations, and is not independently preregistered. For AZT1D transfer, three DDATSHR validation participants were selected by a fixed participant-ID hash, leaving 14 training participants. Six base methods and three seeds used the primary tuning budget and training-only scales, with equal-participant validation MAE; the stack minimized patient-balanced validation squared error. All 18 checkpoints, scales, weights, code and protocol were hashed and frozen before external scoring, with no AZT1D tuning or calibration. RF robustness additionally refit all 46 primary outer folds, two feature conditions and three seeds with three inner validation participants (276 fits). This changes both validation coverage and training-set size. Chronos-2 version 2.3.2, model revision 29ec3766d36d6f73f0696f85560a422f50e8498c, used nine glucose observations, median forecasts and cross-learning disabled. DDATSHR uses predicted steps 2/4/6 and delayed DiaData steps 3/5/7. Primary CGM-only comparators also retain time-of-day features; consequently this is a matched-origin, not perfectly feature-identical, comparison. Pretraining overlap is unknown. Additional analyses are descriptive and do not enlarge or replace the primary 18-test family.

Research assistance: OpenAI Codex was used during the September 2026 revision for repository and literature inspection, analysis-code development, experiment execution, result interpretation, figure-generation code and manuscript drafting. The precise hosted service build was not archived. The experimental local generator is specified separately in Section 6.1. AI assistance is not independent peer review; the responsible authors must review the entire work before submission [16].

## 6. Zero/Few-Shot On-Premises LLM Explainability

### 6.1 Rationale for removing fine-tuning

The original goal of avoiding task-specific LLM fine-tuning is retained. Zero-shot generation receives the declared field and tier rules; few-shot generation additionally receives three fixed development examples. No fine-tuning is performed in either arm. There is no paired fine-tuned comparator, so equivalence or training-cost savings relative to an unverified historical run are not inferred.

Case selection uses fixed random sampling of at most one low, in-range and high latest-input state per participant, independently of forecast error. A hash-ordered participant split reserves approximately one fifth for development; missing states are not fabricated. The remaining participants form the final test set. This partition controls narrative-case exposure; the RF models retain their per-patient forecasting folds rather than a single globally held-out narrative cohort. Three development examples, where available across states, define the few-shot arm. The detector and prompt are frozen before final generation. This stratified case set measures consistency across sampled states, not prevalence-weighted clinical performance.

The local model is DeepSeek-V4-Flash-0731, revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 [17], served by vLLM with tensor parallelism four, an 8192-token context, FP8 KV cache, eager execution and GPU-memory utilization 0.55. Temperature is zero, seed is 2026, the output budget is 2048 tokens, and thinking is disabled. All generation requests use loopback; no patient case is sent to a hosted inference API. The forecasting and generation processes share the available four H200 GPUs. Reported request times therefore reflect the recorded concurrent workload, including queueing, rather than isolated hardware latency or a real-time service guarantee. A separate sequential warm processing replay is described in Section 7.5.

### 6.2 Safety filters

The validator requires all declared fields and compares each against its own source value after predefined rounding. Unknown fields, duplicate keys, incorrect types, nonfinite values and missing claims are rejected. No sentence is exempted because it mentions a guideline, and HR cannot ground a glucose claim. Only validated fields reach a controlled-language renderer. Rejection produces a deterministic source template and is counted as a rejection of the LLM completion. Free-form clinical prose is not certified or released by this path. Thus acceptance measures complete contract preservation, not clinician-rated explanation quality or the value of unrestricted natural language.

A separately specified synthetic suite defines gold labels by mutation operation. Development families cover field values, units, horizons, tiers and omissions; separately designated challenge families include SHAP signs, rate intervals, strings, extra intervals/advice, Unicode keys, booleans, duplicate keys, empty objects and nonfinite values. Valid controls test formatting variation. A distinct boundary suite corrupts an input and then copies it faithfully, deliberately testing what source-relative verification cannot establish. Cases within a mutation family are related; aggregate counts are descriptive rather than independent clinical-trial observations. The same implementation process designed and inspected the detector and synthetic suite; this is a transparent contract regression audit, not an independently blinded adversarial evaluation. The suite is not FActScore or clinician annotation [18].

### 6.3 Evaluation and results

The fixed selection contained 135 cases from 46 people; 111 cases from 38 people formed the final test set. Development used 24 cases from 8 other people. Zero-shot and few-shot complete-contract acceptance was 100/111 and 96/111, respectively. The deterministic arm supplies a direct comparator with no language-model inference. All raw responses and request failures are retained, including rejected outputs.

**Table 2.** Final-test generation contract acceptance and request latency. Template fallback is excluded from LLM acceptance. Latency measures the request (template serialization for the deterministic arm), including concurrent workload and queueing but excluding validation and rendering. Values below 0.001 s round to 0.000.

| Arm | Cases | Accepted | Acceptance | Request errors | Median s | p95 s |
| --- | --- | --- | --- | --- | --- | --- |
| deterministic | 111 | 111 | 100.0% | 0 | 0.000 | 0.000 |
| zero_shot | 111 | 100 | 90.1% | 0 | 27.128 | 27.614 |
| few_shot | 111 | 96 | 86.5% | 0 | 27.206 | 27.606 |

Giving each test participant equal weight, contract acceptance was zero_shot: 89.9% (95% interval 83.8–95.2%); few_shot: 86.0% (95% interval 78.9–92.1%). Intervals use 10,000 participant bootstrap resamples and are descriptive; three state-selected cases per person are not independent patients.

Rejected final completions included SHAP mismatches in 2 completions; direction mismatches in 25 completions; other_contract mismatches in 2 completions; tier mismatches in 1 completions. Categories can overlap within a completion. The prompt and detector were not modified in response to these final-test failures.

The separately designated challenge families produced 300 invalid contracts; 300 were rejected. Among 60 valid controls, 0 were rejected. All 30 source-corruption boundary cases were accepted despite deliberately wrong underlying HR values. Thus zero observed within-contract false negatives would not imply zero end-to-end factual errors. The development-family results are reported separately in the supplement.

Figure 2 separates acceptance of generated contracts from detector rejection by suite. Passing this suite cannot establish absence of untested failure modes. In particular, source-corruption cases demonstrate that copying an inaccurate input can preserve every declared field. Forecast error, misleading model attribution and clinical relevance remain outside the contract's verification scope.

## 7. Experiments and Results

### 7.1 Original OhioT1DM results (prior work, not independently verified)

The original article [2] remains the contextual predecessor, not a numerical control group. Its six-modal weights, reconstructed streams and evaluation predictions are unavailable in this repository. Historical accuracy, fine-tuning cost and emergency-alert scores are therefore not treated as reproduced results. The current study cannot identify a modality effect by subtracting its errors from the earlier reported OhioT1DM errors.

### 7.2 External-cohort evaluation on DDATSHR

In each outer fold, one participant is withheld for testing. A deterministic seeded shuffle, based on the held-out identifier, assigns one other participant to inner validation; all remaining participants fit the base models. The validation participant is never included in base fitting. All architectures and seeds share these allocations. This is a nested-holdout design, not full nested cross-validation. Models are not refitted on the validation participant after selection, preserving comparable fitting sets for ensemble members and standalone models.

This patient-held-out retraining protocol evaluates the approach on the external adolescent cohort; it does not test transfer of frozen original weights. The corrected primary analysis includes 17 adolescents and 72,414 origins. RF subject-macro MAE is 18.12/30.54/38.82 mg/dL at 30/60/90 minutes.

Secondary DiaData analysis. The identical within-cohort selection procedure is applied to 29 non-DDATSHR DiaData participants and 14,500 origins. RF subject-macro MAE is 23.92/34.25/41.63 mg/dL at nominal 30/60/90 minutes. Its extra input delay and source composition differ from DDATSHR, so absolute cross-cohort scores do not isolate age or hardware effects.

Table 3 and Figure 3 compare the primary methods. The lowest point-estimate MAEs were ddatshr at 30 minutes: Stack (17.11 mg/dL); ddatshr at 60 minutes: Stack (29.61 mg/dL); ddatshr at 90 minutes: TFTEncoder (38.11 mg/dL); diadata at 30 minutes: Ridge (22.07 mg/dL); diadata at 60 minutes: Ridge (33.48 mg/dL); diadata at 90 minutes: TFTEncoder (40.69 mg/dL). These rankings are descriptive, not multiplicity-adjusted pairwise superiority claims.

**Table 3.** Subject-macro MAE (mg/dL) at 30/60/90 minutes; seed-averaged predictions for learned methods.

| Cohort | Model | 30 min | 60 min | 90 min |
| --- | --- | --- | --- | --- |
| ddatshr | Persistence | 20.65 | 34.25 | 44.16 |
| ddatshr | LinearTrend | 22.75 | 48.91 | 75.61 |
| ddatshr | Ridge | 17.87 | 31.44 | 40.52 |
| ddatshr | RandomForest | 18.12 | 30.54 | 38.82 |
| ddatshr | XGBoost | 17.62 | 30.42 | 38.93 |
| ddatshr | TFTEncoder | 17.21 | 29.68 | 38.11 |
| ddatshr | BiLSTMTransformer | 17.37 | 30.08 | 38.62 |
| ddatshr | CNNGRU | 17.26 | 29.85 | 38.30 |
| ddatshr | Stack | 17.11 | 29.61 | 38.13 |
| ddatshr | TreeStack | 17.67 | 30.38 | 38.78 |
| diadata | Persistence | 25.32 | 36.18 | 44.41 |
| diadata | LinearTrend | 28.83 | 51.69 | 74.77 |
| diadata | Ridge | 22.07 | 33.48 | 41.26 |
| diadata | RandomForest | 23.92 | 34.25 | 41.63 |
| diadata | XGBoost | 23.82 | 34.95 | 42.54 |
| diadata | TFTEncoder | 23.51 | 33.62 | 40.69 |
| diadata | BiLSTMTransformer | 24.86 | 35.71 | 43.18 |
| diadata | CNNGRU | 23.89 | 33.85 | 41.06 |
| diadata | Stack | 23.48 | 33.82 | 41.15 |
| diadata | TreeStack | 23.74 | 34.25 | 41.73 |

### 7.3 Per-Patient Robustness

DDATSHR RF 30-minute participant RMSE ranged from 18.97 to 28.90 mg/dL. Participant 907 had RMSE 22.75 mg/dL (rank 13 of 17, highest error first), 1,305 eligible origins and 21.4 recorded days. Its source-grid glucose and HR missing fractions were 16.3% and 7.2%, and eligible-input glucose SD was 73.27 mg/dL. Actual sensor labels were Abbott and Medtronic. These measurements replace the earlier inference from a small, device-selected sample; they do not identify a physiological cause. Exploratory Spearman associations with participant RMSE were ddatshr: recent absolute glucose rate rho=0.92, glucose SD rho=0.28, glucose missing fraction rho=0.13, HR missing fraction rho=0.02; diadata: recent absolute glucose rate rho=0.86, glucose SD rho=0.80, glucose missing fraction rho=-0.10, HR missing fraction rho=0.05. These cross-participant associations are unadjusted, may reflect source composition and do not establish causes of an individual participant’s error.

Figure 4 displays patient heterogeneity, and Figure 5 shows the RF 30-minute Clarke distributions. Numerical zone summaries use all pairs; plot axes have explicit display limits. Zone A alone is not a complete error characterization: potentially important C, D and E errors remain visible in the full table and are not absorbed into a single A+B percentage.

### 7.4 Statistical Significance

For patient p with n_p evaluated origins, let e(p,t,h) = Y(p,t,h) − ŷ(p,t,h). The familiar glucose-prediction metrics are retained with patient and horizon indices made explicit:

  RMSE(p,h) = sqrt[(1/n_p) Σ_t e(p,t,h)²].     (4)

  MAE(p,h) = (1/n_p) Σ_t |e(p,t,h)|.     (5)

  MARD(p,h) = (100/n_p) Σ_t |e(p,t,h)| / Y(p,t,h).     (6)

Subject-macro MAE averages Equation (5) over participants, giving each person equal weight. Pooled metrics instead combine all evaluated origins; the two aggregations must not be interchanged. Supplementary pooled tables retain MARD alongside MAE/RMSE.

The primary outcome is the mean of participant-specific MAEs. Pooled MAE/RMSE, mean participant RMSE and all five Clarke error-grid zones are secondary [19]. Clarke analysis is descriptive because its original application was contemporaneous measurement accuracy, not treatment decisions based on forecasts. The lower-C boundary is correctly limited to reference values from 130 to 180 mg/dL. We also report errors for targets below 70, between 70 and 180, and above 180 mg/dL, with target-level sensitivity and precision. These are not prospective episode-alert metrics.

For each cohort and horizon, the three protocol-specified paired comparisons are RF with HR versus RF without HR, stack versus RF, and RF versus persistence. The effect is the mean paired subject-MAE difference; negative values favor the first method. Percentile intervals resample participants 10,000 times using a fixed seed. Two-sided Wilcoxon signed-rank tests use the Pratt zero convention and Holm correction across all 18 comparisons. The signed-rank test concerns a paired location shift, whereas the reported effect is the mean difference; these are not identical estimands. Bootstrap intervals are unadjusted descriptive intervals, so crossing zero is not substituted for the Holm decision. No window-level significance test is used.

These inferences are conditional on the trained models, observed participants and shared training sets. Overlapping outer training folds and convenience sampling limit population-level interpretation, and seed repetitions do not increase the number of independent participants. For descriptive uncertainty intervals around forecasts, the 95th percentile of absolute residuals from the inner participant's seed-averaged predictions defines a symmetric radius. Coverage and width are measured on the outer patient. This is not a distribution-free calibration guarantee, and such intervals are not sent to the language model.

RF HR − RF CGM showed a Holm-significant improvement in 0/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6; Stack − RF showed a Holm-significant improvement in 3/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6; RF − persistence showed a Holm-significant improvement in 4/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6. Failure to reject is not evidence of equivalence.

Figure 6 displays the paired contrasts.

**Table 4.** Paired subject-MAE differences, unadjusted patient-bootstrap 95% intervals and Holm-adjusted p values.

| Cohort | Horizon | Contrast | MAE difference | 95% interval | Holm p |
| --- | --- | --- | --- | --- | --- |
| ddatshr | 30 | RF HR − RF CGM | +0.22 | [+0.07, +0.39] | 0.1613 |
| ddatshr | 30 | Stack − RF | -1.01 | [-1.24, -0.78] | 0.0002747 |
| ddatshr | 30 | RF − persistence | -2.53 | [-2.89, -2.15] | 0.0002747 |
| ddatshr | 60 | RF HR − RF CGM | +0.15 | [-0.06, +0.41] | 1 |
| ddatshr | 60 | Stack − RF | -0.93 | [-1.26, -0.63] | 0.0004272 |
| ddatshr | 60 | RF − persistence | -3.71 | [-4.57, -2.85] | 0.0002747 |
| ddatshr | 90 | RF HR − RF CGM | +0.12 | [-0.11, +0.43] | 1 |
| ddatshr | 90 | Stack − RF | -0.69 | [-1.13, -0.32] | 0.03352 |
| ddatshr | 90 | RF − persistence | -5.34 | [-6.72, -3.99] | 0.0002747 |
| diadata | 30 | RF HR − RF CGM | +0.11 | [-0.18, +0.44] | 1 |
| diadata | 30 | Stack − RF | -0.44 | [-0.87, +0.03] | 0.08889 |
| diadata | 30 | RF − persistence | -1.40 | [-2.33, -0.47] | 0.03352 |
| diadata | 60 | RF HR − RF CGM | -0.08 | [-0.59, +0.52] | 0.8843 |
| diadata | 60 | Stack − RF | -0.43 | [-1.13, +0.35] | 0.8843 |
| diadata | 60 | RF − persistence | -1.93 | [-3.35, -0.48] | 0.06594 |
| diadata | 90 | RF HR − RF CGM | -0.34 | [-0.98, +0.44] | 0.6142 |
| diadata | 90 | Stack − RF | -0.48 | [-1.42, +0.50] | 0.8843 |
| diadata | 90 | RF − persistence | -2.78 | [-4.72, -0.84] | 0.08008 |

Across individual RF training seeds, 30-minute subject-macro MAE ranged 18.12–18.15 mg/dL for ddatshr; 23.92–23.95 mg/dL for diadata. Full seed-specific results are supplied, rather than treating seeds as additional patients.

**Table 5.** RF pooled RMSE and all Clarke zones (A/B/C/D/E, %).

| Cohort | Horizon | Pooled RMSE | Clarke A / B / C / D / E (%) |
| --- | --- | --- | --- |
| ddatshr | 30 | 24.97 | 81.38 / 15.89 / 0.06 / 2.67 / 0.00 |
| ddatshr | 60 | 41.09 | 63.32 / 31.74 / 0.65 / 4.20 / 0.09 |
| ddatshr | 90 | 51.08 | 53.36 / 39.47 / 1.11 / 5.78 / 0.28 |
| diadata | 30 | 26.02 | 76.99 / 19.04 / 0.09 / 3.88 / 0.00 |
| diadata | 60 | 36.82 | 62.83 / 30.63 / 0.34 / 6.14 / 0.06 |
| diadata | 90 | 44.46 | 53.79 / 37.64 / 0.55 / 7.87 / 0.15 |

Descriptive RF validation-residual interval coverage was ddatshr: 94.2%/94.2%/94.2% at 30/60/90 minutes (mean widths 103.3/171.1/210.9 mg/dL); diadata: 90.9%/90.8%/90.5% at 30/60/90 minutes (mean widths 136.3/195.0/238.1 mg/dL). These observed coverages are not assumed to equal their nominal 95% level.

### 7.5 Relationship to the Original OhioT1DM Results

After correction across 18 tests, adding HR improved RF in 0/6 cohort–horizon comparisons, while stacking improved RF in 3/6. The stack findings were ddatshr: 3/3 horizons; diadata: 0/3 horizons.

Retraining has now made the model comparison measurable on matched samples. The earlier claim that all deep models and the stack underperform RF is withdrawn. The corrected findings do not establish a universal preference for either trees or deep models; nor do they identify data scarcity or missing modalities as the cause of a historical cross-study gap.

Four-hour history sensitivity uses Ridge and XGBoost at seed 42. Both two- and four-hour versions are refitted on the four-hour eligible patient/timestamp manifest, with identical split allocations. Separate eligibility losses are reported. Source transfer trains RF and XGBoost in each direction between HUPA-UCM and D1NAMO, reserving one training-source participant for validation, and averages three seeds. These are exploratory checks rather than additional confirmatory tests.

Per-participant error is compared descriptively with recording length, glucose and HR missing fractions, glucose mean/variability, recent absolute rate of change and available DDATSHR demographics. Missing fractions concern the source grid; glucose distribution summaries concern eligible inputs. Spearman associations are exploratory and do not establish physiology as the cause of error. Participant 907 is examined without assuming that the earlier outlier ranking survives corrected sampling.

The four-hour requirement retained 17 DDATSHR and 26 DiaData participants. The two history versions use the same longer-history eligible origins and participant allocations. Source transfer tests both directions between the seven primary D1NAMO and 22 HUPA-UCM participants; its training-source sizes are consequently unequal.

**Table 6.** Exploratory history and source-transfer results; these do not enter the primary significance family.

| Exploratory condition | Model | Patients | Macro MAE 30/60/90 |
| --- | --- | --- | --- |
| ddatshr, 120 min history | Ridge | 17 | 18.38/32.36/41.71 |
| ddatshr, 240 min history | Ridge | 17 | 18.35/32.26/41.55 |
| ddatshr, 120 min history | XGBoost | 17 | 18.27/31.61/40.60 |
| ddatshr, 240 min history | XGBoost | 17 | 18.31/31.69/40.73 |
| diadata, 120 min history | Ridge | 26 | 21.91/33.54/42.05 |
| diadata, 240 min history | Ridge | 26 | 22.01/33.69/42.37 |
| diadata, 120 min history | XGBoost | 26 | 24.42/36.41/44.65 |
| diadata, 240 min history | XGBoost | 26 | 24.67/36.30/44.68 |
| Held-out source: D1NAMO | RandomForest | 7 | 27.41/39.16/46.11 |
| Held-out source: D1NAMO | XGBoost | 7 | 30.72/43.82/49.92 |
| Held-out source: HUPA-UCM | RandomForest | 22 | 27.22/40.30/48.35 |
| Held-out source: HUPA-UCM | XGBoost | 22 | 28.29/42.66/52.86 |

Additional frozen-model and baseline evidence is summarized in Table 7 and Figure 7. AZT1D’s lowest point-estimate MAEs were 30 minutes: Stack, 15.06 mg/dL; 60 minutes: BiLSTMTransformer, 24.33 mg/dL; 90 minutes: BiLSTMTransformer, 30.34 mg/dL. Chronos-2 MAE on ddatshr was 19.70/34.33/45.04 mg/dL. Three-validator RF HR-minus-CGM differences on ddatshr were +0.17/+0.09/+0.05 mg/dL. Chronos-2 MAE on diadata was 23.85/36.23/45.37 mg/dL. Three-validator RF HR-minus-CGM differences on diadata were +0.21/+0.15/+0.10 mg/dL. These point estimates do not establish universal architecture superiority or HR equivalence. The supplementary tables retain all seeds, participant errors, glycemic ranges, Clarke zones and paired descriptive intervals.

**Table 7.** Frozen DDATSHR glucose-only models evaluated on AZT1D. Three-seed mean predictions; participant-macro MAE (mg/dL), with descriptive 95% participant-bootstrap intervals. No model was selected using AZT1D results.

| Model | 30 min | 60 min | 90 min |
| --- | --- | --- | --- |
| Persistence | 17.23 [16.23, 18.25] | 28.18 [26.47, 29.90] | 35.87 [33.59, 38.20] |
| LinearTrend | 20.68 [19.38, 22.06] | 42.74 [40.18, 45.38] | 65.68 [61.86, 69.62] |
| Ridge | 15.96 [15.11, 16.83] | 25.86 [24.63, 27.10] | 32.00 [30.46, 33.59] |
| RandomForest | 15.22 [14.39, 16.07] | 25.05 [23.78, 26.33] | 31.34 [29.80, 32.94] |
| XGBoost | 15.24 [14.41, 16.10] | 24.95 [23.68, 26.23] | 31.22 [29.66, 32.84] |
| TFTEncoder | 15.25 [14.38, 16.16] | 24.55 [23.19, 25.97] | 30.73 [28.97, 32.57] |
| BiLSTMTransformer | 15.18 [14.32, 16.09] | 24.33 [22.99, 25.70] | 30.34 [28.66, 32.07] |
| CNNGRU | 15.35 [14.47, 16.26] | 24.64 [23.29, 26.02] | 30.63 [28.93, 32.40] |
| Stack | 15.06 [14.22, 15.93] | 24.79 [23.51, 26.08] | 31.12 [29.54, 32.74] |


The processing replay selected 30 test cases by a fixed hash with representation from all three primary sources, before timings were observed. Each arm recomputed the loaded RF forecast and genuine additive TreeSHAP, then generated, validated and rendered output; three warm-up runs were excluded. Requests ran sequentially with rotated arm order after GPU forecasting finished, using the frozen prompt/model/detector and a 600-second request timeout. Available histories and models were already in memory; default server prefix caching remained enabled. CPU-only audit and document tasks continued during part of the replay, so it was not a dedicated idle-machine benchmark. This measures warm processing, excluding device acquisition, network ingestion, startup and clinical workflow. The soft processing deadline is the next 15-minute sensor update. The 30 cases represented 24 participants. Median total processing times were deterministic: 0.124 s; few_shot: 26.659 s; zero_shot: 26.600 s. There were 0 observed deadline misses and 0 request errors across 90 measured runs. This small warm replay cannot establish hard real-time guarantees or patient benefit.

**Table 8.** Sequential processing replay, 30 cases per arm. Total includes prediction, TreeSHAP, generation, validation and rendering. Acceptance remains a source-relative contract measure; failures and fallbacks remain in latency denominators.

| Arm | Accepted | Median (s) | 95th percentile (s) | Maximum (s) | Deadline misses |
| --- | --- | --- | --- | --- | --- |
| deterministic | 30/30 | 0.124 | 0.413 | 0.449 | 0 |
| few_shot | 25/30 | 26.659 | 26.943 | 26.982 | 0 |
| zero_shot | 26/30 | 26.600 | 26.919 | 37.552 | 0 |

## 8. Discussion

**External-cohort generalization and the modality discrepancy.**

After correction across 18 tests, adding HR improved RF in 0/6 cohort–horizon comparisons, while stacking improved RF in 3/6. The stack findings were ddatshr: 3/3 horizons; diadata: 0/3 horizons. RF HR − RF CGM showed a Holm-significant improvement in 0/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6; Stack − RF showed a Holm-significant improvement in 3/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6; RF − persistence showed a Holm-significant improvement in 4/6 cohort–horizon comparisons and a Holm-significant deterioration in 0/6. Failure to reject is not evidence of equivalence.

The matched design resolves a basic comparison problem: tree and neural results now refer to the same participants and targets. It does not prove that a particular model class is universally preferable. Small search grids, one inner participant, different training objectives, reduced inputs and complete-case selection remain part of the experiment. A worse neural result would not identify data scarcity as its cause; a better neural result would not verify the earlier unavailable multimodal model.

The sensor-aware retention result is itself methodologically consequential. A five-minute complete-case requirement on a naturally 15-minute stream can select devices and people before model training. Correcting this changes the study population as well as the amount of data. Consequently, historical scores from the earlier pipeline are not pooled with the corrected scores, and improvements across pipeline versions are not attributed solely to the new model fitting procedure.

The HR ablation estimates conditional added predictive information in windows with observed HR. It does not compare a complete six-modal clinical system against a two-modal system. There are no paired observations of HRV, oxygen saturation, blood pressure and sleep for this cohort, so their missing contribution cannot be estimated by subtraction from historical OhioT1DM errors. Source transfer and the matched history sensitivity address narrower questions for which the data supply an actual comparison.

Participant-level distributions are more informative than selecting one outlier as a physiological case study. Patient 907's recorded duration, eligible samples, missingness, sensor composition and glucose variability can be measured. Insulin timing, meals, sensor placement and activity context are not uniformly available in the fitted schema; associations involving available demographics cannot separate these explanations. We therefore avoid labelling high forecast error as a physiological phenotype or attributing it to insufficient data length alone.

**Zero/few-shot explanation and filter false negatives.**

The deterministic comparator is essential: every declared output can already be rendered from structured values. Any claim that a language model improves usability or clinical explanation would need additional evidence, such as blinded clinician and patient evaluation. Contract acceptance and copying latency cannot supply that evidence. The present experiment measures reliability of an optional language-model step under strict constraints; it does not establish that the step is necessary.

Fail-closed typed validation removes the tested pooled-number and vacuous-pass failure modes by narrowing what is released. This costs expressiveness: unrecognized but potentially correct prose is rejected along with incorrect content. The study reports the resulting acceptance coverage rather than counting fallback delivery as successful LLM reasoning. The guarantee is limited to consistency with the supplied schema and values. It neither authenticates source measurements nor validates forecasting accuracy, SHAP semantics, study-score calibration or treatment safety.

The earlier claim of perfect numerical grounding with weak tier assignment depended on a small, single-patient sample and an inadequate numerical detector. The present results do not preserve that claim merely for narrative continuity. No paired fine-tuned comparator, clinician rating or prospective alert study is available; zero-shot/few-shot operation therefore cannot be described as equivalent to fine-tuning or as a validated lower-cost clinical replacement. The large local model and shared four-GPU service also have infrastructure costs, despite requiring no task-specific parameter updates.

### 8.1 Limitations

The primary cohorts contain only 17 and 29 evaluable participants, with substantial overlap among windows and incomplete representation of source populations. Excluding missing inputs and targets may preferentially retain easier periods. DiaData's prior timestamp processing limits claims about exact physiological lead time despite the conservative input delay. Source-held-out tests involve just two retained sources, and source identity combines multiple demographic and acquisition differences. The additional 24-person AZT1D AID cohort evaluates glucose-only transfer and excludes one ambiguous measurement field. Its lower absolute errors cannot isolate age, therapy, sensor or model effects across cohorts.

One inner participant makes primary hyperparameter and ensemble-weight estimates variable; the three-participant RF sensitivity does not remove this limitation for every model. Three seeds characterize some optimization variability, not uncertainty over all plausible training sets. Validation-residual intervals can fail under between-person shift. The tested neural variants and small tuning budget are not an exhaustive comparison of foundation models or every specialized architecture, despite the additional Chronos-2 baseline. Its pretraining overlap is unknown and nine-point contexts may disadvantage a foundation model; no superiority claim generalizes beyond this setting. There is no evidence here of prospective clinical benefit, clinical acceptability or safer insulin dosing. The generation challenge is synthetic and machine-labelled, with no independent clinician adjudication. External replication remains necessary.

The title retains the original framework and adolescent-cohort goal. Its external-cohort evaluation uses retraining, and its real-time use remains a deployment objective. The new AZT1D test supports frozen newly trained glucose-only model transfer, not frozen historical multimodal model validation in adolescents. Warm processing replay does not demonstrate a live clinical explanation service or an emergency-response system. These distinctions remain potential reviewer concerns even with complete disclosure.

## 9. Conclusions and Future Work

The paper retains its original research goal: assess GlucoSense forecasting in an independent adolescent cohort and examine local zero/few-shot explainability without task-specific fine-tuning. After correction across 18 tests, adding HR improved RF in 0/6 cohort–horizon comparisons, while stacking improved RF in 3/6. The stack findings were ddatshr: 3/3 horizons; diadata: 0/3 horizons. Typed generation accepted 100/111 zero-shot and 96/111 few-shot test completions, while explicit source-corruption tests remained undetectable to a source-relative checker.

All three reduced-input deep models have now been evaluated on the same targets as the baselines. The optional LLM step remains bounded by source-relative verification and is not shown superior to deterministic rendering. The additional AZT1D test evaluates frozen glucose-only transfer and the replay measures warm processing. Future work should extend frozen multimodal validation, assess live sensing-to-output deadlines, and obtain independent clinical or user evaluation before claiming clinical utility. No clinical trial was conducted or is claimed in this retrospective computational study.

### 9.1 Reproducibility and research assistance

The supplement provides source acquisition, preparation, training, scoring, figure and manuscript-generation code; pinned dependencies; protocol and source hashes; per-patient aggregate results; and artifact integrity checks. Checkpoints, exact case inputs and per-origin predictions are retained locally, with regeneration instructions tied to the public source releases. The analysis plan was recorded during this revision before corrected experiments, but was not independently preregistered. Smoke tests were used to check implementation and resource use; they are excluded from reported results. Secondary analyses are explicitly exploratory.

OpenAI Codex assisted repository inspection, literature searching, code development, experiment orchestration, analysis and manuscript drafting. The local DeepSeek model generated the experimental outputs. No synthetic clinical measurements replace the observed forecasting data; synthetic records are confined to the labelled detector challenge. Human authors must review the resulting code, results and manuscript and assume responsibility before submission. Research-assistance disclosure follows the substantive-use distinction in MDPI policy [16].

## Supplementary Materials

The revision evidence archive contains the complete executable analysis, model/configuration manifests, per-patient and pooled result tables, seed and ablation outputs, detector challenges, prompt specifications, source provenance and vector/raster figures. Large checkpoints and participant-level artifacts can be regenerated from the pinned public sources using the included commands. The audit record identifies any remaining author-only submission requirements.

## Author Contributions

Authorship and affiliation are retained from the supplied draft. The named authors must confirm the final CRediT allocation and approve the complete revised work before submission; no author sign-off is inferred from automated execution.

## Funding

The authors must confirm the applicable funding statement and any grant identifiers before submission. No funding declaration has been independently established by this computational revision.

## Institutional Review Board Statement

This analysis reuses publicly distributed, de-identified research datasets and introduces no new participant recruitment or intervention. The authors must confirm their institution's secondary-use review or exemption determination and the appropriate wording before submission. This draft does not invent an approval number or exemption.

## Informed Consent Statement

Data collection and consent belong to the original source studies. The authors must confirm that the planned analysis and dissemination comply with the applicable source terms and institutional requirements; no new consent was obtained in this revision.

## Data Availability Statement

Original data are distributed by the DDATSHR repository, DiaData release and AZT1D release cited in [1, 10, 15]. This work does not re-host their raw participant records. Acquisition scripts verify the pinned source hashes. Code, aggregate results, protocol, detector tests and reproduction instructions are prepared in the accompanying revision evidence archive. The project repository is https://github.com/hossain0707/glucosense-ai-extended; access depends on its repository permissions. Public accessibility has not been established, so the supplement supplies the reviewable code and aggregate evidence independently of that access.

## Acknowledgments

OpenAI Codex (September 2026 revision sessions; exact hosted build not archived) assisted research design refinement, literature searching, software development, execution, analysis and drafting. DeepSeek-V4-Flash-0731 was used for the local generation experiment. The authors' substantive review and acceptance of responsibility must be completed before submission; this draft does not state that such review has already occurred.

## Conflicts of Interest

The authors must confirm the final conflict-of-interest declaration before submission.

## References

1. ICT Innovaties Zorg. Dataset: Diabetes Adolescents Time Series with Heart Rate. https://github.com/ictinnovaties-zorg/dataset-diabetes-adolescents-time-series-with-heart-rate (accessed 12 September 2026; pinned commit in Methods).
2. Hossain, M.N.; Kim, J.-N. Multimodal Deep Learning for Real-Time Glucose Prediction with LLM Explainability. *J. Korean Inst. Convergence Signal Process.* **2026**, *27*, 143–150. https://doi.org/10.23087/jkicsp.2026.27.3.006.
3. Marling, C.; Bunescu, R. The OhioT1DM Dataset for Blood Glucose Level Prediction: Update 2020. *CEUR Workshop Proc.* **2020**, *2675*. https://ceur-ws.org/Vol-2675/paper11.pdf.
4. Kolev, M.; Naskinova, I.; Milev, M.; Stoilova, S.; Nikolova, I. A Comprehensive Benchmark of Machine Learning Methods for Blood Glucose Prediction in Type 1 Diabetes: A Multi-Dataset Evaluation. *Appl. Sci.* **2026**, *16*, 3928. https://doi.org/10.3390/app16083928.
5. Lim, B.; Arık, S.Ö.; Loeff, N.; Pfister, T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. *Int. J. Forecast.* **2021**, *37*, 1748–1764. https://arxiv.org/abs/1912.09363.
6. Chen, T.; Guestrin, C. XGBoost: A Scalable Tree Boosting System. *Proc. KDD* **2016**, 785–794. https://doi.org/10.1145/2939672.2939785.
7. Lundberg, S.M.; et al. From local explanations to global understanding with explainable AI for trees. *Nat. Mach. Intell.* **2020**, *2*, 56–67. https://doi.org/10.1038/s42256-019-0138-9.
8. Lu, B.; Liang, Z.; Pontius, R.; Tang, S.; Prioleau, T. GlucoFM-Bench: Benchmarking Time-Series Foundation Models for Blood Glucose Forecasting. *arXiv* **2026**, arXiv:2606.06881, version 1. https://arxiv.org/abs/2606.06881. Preprint.
9. Ansari, A.F.; et al. Chronos-2: From Univariate to Universal Forecasting. *arXiv* **2025**, arXiv:2510.15821. https://arxiv.org/abs/2510.15821. Model and implementation: https://github.com/amazon-science/chronos-forecasting.
10. Cinar, B.; Maleshkova, M. DiaData: An Integrated Large Dataset for Type 1 Diabetes and Hypoglycemia Research. *BIO Web Conf.* **2025**, *195*, 03001. https://doi.org/10.1051/bioconf/202519503001. Data release: https://doi.org/10.5281/zenodo.17285631.
11. Cinar, B.; Maleshkova, M. Benchmarking Hypoglycemia Classification Using Quality-Enhanced DiaData. *IEEE J. Biomed. Health Inform.* **2025**, *29*, 8831–8838. https://doi.org/10.1109/JBHI.2025.3620603.
12. Gupta, V.; Grensing, F.; Cinar, B.; van den Boom, L.; Maleshkova, M. Towards Improved Short-term Hypoglycemia Prediction and Diabetes Management based on Refined Heart Rate Data. *arXiv* **2026**, arXiv:2603.20345. https://arxiv.org/abs/2603.20345. Preprint.
13. Alredaini, R.; Abulkhair, M.; Almisbahi, H. Interpretable glucose forecasting for type 2 diabetes across traditional, deep, and large language models. *Sci. Rep.* **2026**, *16*, 2421; published online 16 December 2025. https://doi.org/10.1038/s41598-025-32373-4.
14. Mamun, A.; Arefeen, A.; Racette, S.B.; Sears, D.D.; Whisner, C.M.; Buman, M.P.; Ghasemzadeh, H. LLM-Powered Prediction of Hyperglycemia and Discovery of Behavioral Treatment Pathways from Wearables and Diet. *Sensors* **2025**, *25*, 5372. https://doi.org/10.3390/s25175372. Author preprint: https://arxiv.org/abs/2503.03935.
15. Khamesian, S.; Arefeen, A.; Thompson, B.M.; Grando, M.A.; Ghasemzadeh, H. AZT1D: A Real-World Dataset for Type 1 Diabetes. *Mendeley Data* **2025**, version 1. https://doi.org/10.17632/gk9m674wcx.1. Dataset report: https://arxiv.org/abs/2506.14789.
16. MDPI. Research and Publication Ethics. https://www.mdpi.com/ethics (indexed official policy checked 13 September 2026).
17. DeepSeek-AI. DeepSeek-V4-Flash-0731 model card and local encoding documentation. https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 (accessed 12 September 2026; revision in Methods).
18. Min, S.; et al. FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation. *EMNLP* **2023**. https://aclanthology.org/2023.emnlp-main.741/.
19. Clarke, W.L.; Cox, D.; Gonder-Frederick, L.A.; Carter, W.; Pohl, S.L. Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose. *Diabetes Care* **1987**, *10*, 622–628. https://doi.org/10.2337/diacare.10.5.622.
