# Third-party data and models

This repository contains code only. It does not redistribute source audio,
datasets, model weights, or model outputs. Reproducing the experiments requires
obtaining each resource from its upstream source under its own terms.

| Resource | Role | Upstream source | Terms |
|---|---|---|---|
| MUSDB18-HQ | vocal and accompaniment stems | Zenodo record 3338373 | academic use only; per-track Creative Commons BY-NC-SA (3.0/4.0); access request required |
| MUSDB18 lyrics extension | line-level lyric annotations and source text for lyric references | Zenodo record 3989267 | Creative Commons BY-NC-SA 4.0 |
| LibriSpeech `test-clean` | read-speech source and speech references | OpenSLR 12 | Creative Commons BY 4.0 |
| Whisper large-v3 | ASR backend | `openai/whisper-large-v3` | Apache-2.0 |
| Qwen3-ASR-1.7B | ASR backend | `Qwen/Qwen3-ASR-1.7B` | upstream model card terms; the Qwen3-ASR reference implementation is Apache-2.0 |
| Qwen3-ForcedAligner-0.6B | word timings for the Qwen3-ASR backend | `Qwen/Qwen3-ForcedAligner-0.6B` | upstream model card terms |
| `qwen-asr` | Python runtime used by `tools/qwen3_asr_service.py` | PyPI `qwen-asr`, from QwenLM/Qwen3-ASR | Apache-2.0 |
| NVIDIA Parakeet CTC 1.1B | ASR backend and common forced aligner | `nvidia/parakeet-ctc-1.1b` | Creative Commons BY 4.0 |
| DeepFilterNet2 0.5.6 | tested enhancer in the front-end experiment | Rikorose/DeepFilterNet | MIT OR Apache-2.0 |

MUSDB18-HQ is the strictest of these. It is restricted to academic,
non-commercial use and is not redistributed here, nor are any stems or mixtures
derived from it. Check the upstream terms before publishing anything built from
it.

The primary word-onset CSV annotations used by the paper are derived inputs and
are not redistributed in this repository. `source_capture.preprocessing.musdb_lyrics`
converts a separately obtained annotation bundle into the required JSONL
manifest. The replication word timings can be regenerated from the public
line-level lyrics extension with `source_capture.preprocessing.replication_lyrics`.

Model revisions are not downloaded by this repository. Record the upstream
commit or revision hash you use in your local configuration. The primary
dataset manifest records the Parakeet revision used for the common alignment.
