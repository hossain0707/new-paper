# Source terms and redistribution decisions

DDATSHR: ICT Innovaties Zorg, commit 36379d9397d05bb00ac20cf91a5a2b648fecfafe, CC BY 4.0. Source and licence: https://github.com/ictinnovaties-zorg/dataset-diabetes-adolescents-time-series-with-heart-rate/blob/36379d9397d05bb00ac20cf91a5a2b648fecfafe/LICENSE.md

DiaData: Beyza Cinar and Maria Maleshkova, Zenodo record 17285631, CC BY-NC 4.0, https://zenodo.org/records/17285631 . Pinned archive MD5 2c876bf0b8bc9ad6c7823e672287b216. Release metadata was acquired on Spark on 22 October; later local/website access failures do not change the saved licence. Mixed-cohort derived tables retain this noncommercial restriction; do not treat them as unrestricted commercial data. D1NAMO and HUPA-UCM retain their source attribution within DiaData. Original component records are not repackaged here.

AZT1D: Saman Khamesian, Asiful Arefeen, Bithika M. Thompson, Adela Grando and Hassan Ghasemzadeh, version 1, CC BY 4.0, https://data.mendeley.com/datasets/gk9m674wcx/1 . Source licence checked on 23 October.

Derived outputs are transformations: observed-only availability filtering, participant-held-out modelling and aggregate/per-request software audits. They contain no original participant time-series records. Attribution and links above accompany all derived data in this directory; applicable CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) and CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) terms remain in force. No extra rights to source material or model weights are granted. Qwen weights are not redistributed; consult its NVIDIA Open Model Agreement at the pinned model repository.

Excluded from public upload: raw participant records, prepared input arrays, timestamps/case-selection CSV, request/response payloads containing copied participant measurements, row-level prediction files, large checkpoints, credentials, original professor reviews and correspondence, and private historical drafts. Raw case-bearing evidence remains in the checksum-verified private evidence archive; model binaries remain on the authorized Spark. Public hashes and regeneration scripts identify them without releasing their contents. An independent author can audit them locally; this public package does not claim public access to those payloads.

The public participant-profile table omits the source age, sex and HbA1c fields; only derived recording/model summaries are retained. Reacquire source records locally for demographic reanalysis.
