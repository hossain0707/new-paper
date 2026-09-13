# Literature and claim audit — 12 September 2026

This is a targeted narrative literature search, not a systematic review. Searches covered
glucose forecasting with CGM/HR, multi-dataset benchmarks, SHAP plus LLM explanations,
DiaData preprocessing, original architecture papers, and MDPI author requirements.
Publisher pages, author repositories, original papers and metadata were checked; numerical
results from different datasets/splits are not treated as matched baselines.

| Source | Verified relevance | Consequence for this paper |
|---|---|---|
| [DDATSHR source](https://github.com/ictinnovaties-zorg/dataset-diabetes-adolescents-time-series-with-heart-rate) | Abbott and Medtronic sampling and clock issues differ. | Use original offset-bearing timestamps, sensor-aware bins and explicit clock exclusions. |
| [DiaData dataset paper](https://doi.org/10.1051/bioconf/202519503001) and [release 17285631](https://doi.org/10.5281/zenodo.17285631) | Integrated research resource; the public release is not the full dataset. | Cite version and original sources; report only the actually eligible SDBII patients. |
| [Cinar and Maleshkova, JBHI 2025](https://pubmed.ncbi.nlm.nih.gov/41359690/) | Hypoglycemia classification and quality enhancement with IQR and interpolation. | Forecasting with observed targets is a different estimand. Do not use interpolated future targets as raw evidence. |
| [Upstream integration code](https://github.com/Beyza-Cinar/Preprocessing-DiaData/blob/main/Code/data_integration.py) | Rounds source timestamps; raw exports have already undergone integration. | Do not call the export untouched native sensor data. Apply and disclose the delayed-input safeguard. |
| [Kolev et al., Applied Sciences 2026](https://doi.org/10.3390/app16083928) | The Methods explicitly state that reported experiments use physics-based synthetic data because original downloads were unavailable. | Distinguish real observed records here; do not quote their benchmark as evidence of performance on real D1NAMO/HUPA patients. |
| [Alredaini et al., Scientific Reports](https://pmc.ncbi.nlm.nih.gov/articles/PMC12819561/) | Compares traditional/deep/LLM forecasting for T2D and considers SHAP and language explanations. Online 16 December 2025, collection 2026. | Withdraw claims that combining glucose prediction, SHAP and LLMs is new by itself. |
| [GlucoLens, Mamun et al.](https://arxiv.org/abs/2503.03935) | Wearable/food-based postprandial prediction and counterfactual language support; also published as Sensors 25, 5372. | Acknowledge adjacent work while distinguishing the present multi-horizon sensor-only consistency task. |
| [Gupta et al., March 2026 preprint](https://arxiv.org/abs/2603.20345) | Refined HR data for hypoglycemia prediction. | HR availability and processing are active research topics; no first-use claim. Mark preprint status. |
| [Lim et al., TFT](https://arxiv.org/abs/1912.09363) | General multi-horizon architecture, not a glucose-specific benchmark. | Cite architecture only. Name our encoder-only variant accurately. |
| [Mhaskar et al. 2017](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2017.00014/full) | Deep glucose prediction study. | Remove the old unsupported claim of an 8% HRV-driven gain. |
| [Lundberg et al., TreeSHAP](https://doi.org/10.1038/s42256-019-0138-9) | Tree model attribution. | Archive actual attributions and verify additivity; never equate attribution with causation. |
| [Clarke et al. 1987](https://doi.org/10.2337/diacare.10.5.622) | Error grid originated for contemporaneous glucose measurement accuracy. | Correct lower-C boundary; report A–E separately. Forecast grid results do not certify treatment safety. |
| [OhioT1DM documentation](https://ceur-ws.org/Vol-2675/paper11.pdf) | Available streams do not establish the raw wrist PPG reconstruction claimed in the previous manuscript. | The earlier six-modal results cannot be independently verified from this repository. Do not use their numerical gap as an identified modality effect. |
| [Previous GlucoSense article](https://doi.org/10.23087/jkicsp.2026.27.3.006) | The local publisher-formatted PDF is a journal article, volume 27, pages 143–150, 2026. | Disclose related publication, separate new experiments, and remove reused artwork and unverified historical performance from new results. |
| [MDPI ethics](https://www.mdpi.com/ethics) | Originality, transparent methods/data access, accurate authorship and substantive GenAI disclosure are required. | Prepare disclosures; authors must personally verify responsibility and declarations before submission. |

## Corrected scope

The defensible contribution is an empirical, reproducible comparison under sensor-aware,
patient-disjoint evaluation, coupled with an explicit audit of a tightly constrained
generation contract. This is not a new foundation model, proof of clinical safety,
fine-tuning equivalence study, or prospective deployment. Incremental novelty remains
an editorial judgment; a revised title cannot substitute for substantive results.

## Source records

The experiment folder retains source-tree identifiers, downloaded integration code,
Crossref responses where available, the Zenodo record and the archive checksum. The
previous article is also present locally as a PDF, so its metadata did not depend solely
on the unavailable DOI redirect. Full source datasets are linked rather than re-hosted.

## Additional evidence literature check, 13 September 2026

- AZT1D v1, DOI [10.17632/gk9m674wcx.1](https://data.mendeley.com/datasets/gk9m674wcx/1), is a CC BY 4.0 provider release with 25 T1D AID participants. The [primary report](https://arxiv.org/abs/2506.14789) and archive README describe CGM and pump data. We use CGM only, exclude the ambiguous Subject 14 field before scoring, and do not claim an adolescent-only cohort or new clinical trial. Archive/provider hashes and schema decisions are preserved.
- [GlucoFM-Bench](https://arxiv.org/abs/2606.06881), Lu et al., June 2026 v1 preprint, directly limits novelty claims about multi-dataset foundation-model glucose forecasting. Its preprocessing, datasets and evaluation settings differ; its numerical results are not inserted into our comparison table. Our contribution must be assessed as a matched-protocol replication/extension and failure audit.
- [Chronos-2](https://arxiv.org/abs/2510.15821), Ansari et al., October 2025 preprint, provides the additional frozen comparator. The [official implementation](https://github.com/amazon-science/chronos-forecasting) documents cross-learning, which is explicitly disabled. Model revision and package versions are frozen before inference. Nine glucose points and unknown pretraining overlap limit interpretation. The CGM-only RF comparator additionally uses time-of-day, disclosed in Methods and figure captions.
- CITY’s official named-researcher access form was inspected but not submitted. Its data were not downloaded through the exposed storage link. No author identity or email was invented. Open AZT1D was used instead.
- The official Sensors instructions were revisited; direct page retrieval again failed. Previously indexed official policy evidence is retained. No new institutional approval or guarantee of journal compliance is inferred.
