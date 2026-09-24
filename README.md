# Speech-Song Source Selection in ASR

Reproduction code for **When Speech Competes with Song: Onset Position and
Temporal History Shape Source Selection in Single-Output ASR**.

This repository implements the five experiments reported in the paper. It is
organized around the public reproduction workflow. Source audio, model weights,
annotations, and model outputs are not distributed here; see `THIRD_PARTY.md`
for the upstream terms of everything you need to obtain yourself.

## Experiments

| CLI name | Paper experiment | Manipulation |
|---|---|---|
| `position` | Position | equal-budget vocal suppression at different locations |
| `full-mix` | Full-mix | isolated vocal versus the original accompaniment |
| `history` | History | different gain histories with an identical evaluation window |
| `scrambling` | Scrambling | intact versus block-shuffled lyric streams |
| `front-end` | Front-end | oracle onset suppression and DeepFilterNet2 |

Shared implementation lives in `source_capture/core/`; experiment-specific
stimulus construction and analyses live in `source_capture/experiments/`.

## Installation

```bash
python -m pip install -e .
```

This installs the repository itself as the local Python distribution
`speech-song-source-selection` (version `1.0.0`). It provides the
`source_capture` import package and the `source-capture`,
`source-capture-report`, and `source-capture-validate` commands; it is not a
separate package that must be downloaded from PyPI.

The lightweight dependencies cover stimulus generation, scoring, and
statistics. Install the optional model dependencies for the ASR systems you
intend to run:

```bash
python -m pip install -e '.[whisper]'
python -m pip install -e '.[nemo]'
```

`requirements-inference.txt` lists the same stacks with notes on version
sensitivity. Record the versions you actually use: the Whisper word timestamps
that the windowed analyses depend on are requested through
`return_timestamps="word"`, whose behaviour has changed across `transformers`
releases.

## What you must supply

Stimulus preparation, scoring, and statistics run on the dependencies above.
Everything else comes from upstream:

| Resource | Needed for | Where it comes from |
|---|---|---|
| MUSDB18-HQ | the vocal and accompaniment stems | Zenodo record 3338373; academic use, access request |
| MUSDB18 lyrics extension | the lyric references | Zenodo record 3989267 |
| LibriSpeech `test-clean` | the speech source | OpenSLR 12 |
| LibriSpeech word alignments | speech references and windowed scoring | Montreal Forced Aligner output, for example the `gilkeyio/librispeech-alignments` parquet export; converted with `source_capture.preprocessing.librispeech` |
| Whisper large-v3, Qwen3-ASR-1.7B, Parakeet CTC 1.1B | the three recognizers | Hugging Face; pin local copies and record their revisions |
| DeepFilterNet 0.5.6 (`deep-filter-...-unknown-linux-musl`) | the front-end experiment only | the DeepFilterNet release binaries |

Paths for the three recognizers are checked only by the stages that load them.
You can prepare the dataset and generate every mixture before you have a single
checkpoint, and `transcribe` will tell you which path is still a placeholder.

## Reproduction workflow

Copy the example configuration and replace dataset, model, and output paths:

```bash
cp configs/primary.example.yaml configs/primary.local.yaml
```

Prepare the fully crossed speech-vocal pairs once:

```bash
source-capture dataset prepare --config configs/primary.local.yaml
```

Then run each paper experiment through the same public entry point. For the
Position experiment:

```bash
source-capture position generate   --config configs/primary.local.yaml
source-capture position transcribe --config configs/primary.local.yaml --model whisper
source-capture position score      --config configs/primary.local.yaml --model whisper
source-capture position analyze    --config configs/primary.local.yaml
```

Repeat `transcribe` and `score` for `qwen3` and `ctc`. The History experiment
uses the same stages with `history` in place of `position`.

After History and Scrambling are analyzed for Whisper and Qwen3-ASR, assemble
the four reported confirmatory contrasts and Holm-adjusted p-values with:

```bash
source-capture-report \
  --history outputs/primary/temporal_history/summary.json \
  --scrambling outputs/scrambling/summary.json outputs/scrambling/summary.qwen3.json \
  --output outputs/confirmatory_results.json
```

The paths follow each configuration's `project.output_root`.

Copy the experiment-specific configurations:

```bash
cp configs/full_mix.example.yaml configs/full_mix.local.yaml
cp configs/scrambling.example.yaml configs/scrambling.local.yaml
cp configs/front_end.example.yaml configs/front_end.local.yaml
```

Then run every stage for the remaining experiments. The Full-mix and
Scrambling commands perform source scoring while transcribing, so they do not
have a separate `score` stage:

```bash
source-capture full-mix prepare    --config configs/full_mix.local.yaml
source-capture full-mix transcribe --config configs/full_mix.local.yaml --model whisper
source-capture full-mix analyze    --config configs/full_mix.local.yaml --model whisper

source-capture scrambling prepare  --config configs/scrambling.local.yaml
source-capture scrambling transcribe --config configs/scrambling.local.yaml --model whisper
source-capture scrambling analyze    --config configs/scrambling.local.yaml --model whisper

source-capture front-end prepare   --config configs/front_end.local.yaml
source-capture front-end enhance   --config configs/front_end.local.yaml
source-capture front-end transcribe --config configs/front_end.local.yaml --model whisper
source-capture front-end analyze    --config configs/front_end.local.yaml --model whisper
```

Repeat each `transcribe`/`analyze` pair for `qwen3` and `ctc`.

The independent replication set uses three configurations:

| Configuration | Runs |
|---|---|
| `replication.example.yaml` | dataset preparation and the Position experiment on the 26 replication tracks |
| `replication_history.example.yaml` | the History replication: one 2 s speech-to-lyric contrast against its matched static comparator |
| `replication_full_mix.example.yaml` | the Full-mix replication |

`replication.example.yaml` carries only the Position block, so the History
replication goes through its own configuration rather than through `history`.
Each experiment accepts a subset of the stages listed in the top-level `--help`,
and an invalid combination is reported before anything runs.

Create the local replication configurations and run them as follows:

```bash
cp configs/replication.example.yaml configs/replication.local.yaml
cp configs/replication_history.example.yaml configs/replication_history.local.yaml
cp configs/replication_full_mix.example.yaml configs/replication_full_mix.local.yaml

source-capture dataset prepare --config configs/replication.local.yaml
source-capture position generate --config configs/replication.local.yaml
source-capture position transcribe --config configs/replication.local.yaml --model whisper
source-capture position score --config configs/replication.local.yaml --model whisper
source-capture position analyze --config configs/replication.local.yaml

source-capture history prepare --config configs/replication_history.local.yaml
source-capture history transcribe --config configs/replication_history.local.yaml --model whisper
source-capture history align --config configs/replication_history.local.yaml --model whisper
source-capture history analyze --config configs/replication_history.local.yaml --model whisper

source-capture full-mix prepare --config configs/replication_full_mix.local.yaml
source-capture full-mix transcribe --config configs/replication_full_mix.local.yaml --model whisper
source-capture full-mix analyze --config configs/replication_full_mix.local.yaml --model whisper
```

Repeat the model-dependent replication stages for `qwen3` and `ctc`.

## Windowed analysis and word timings

The History contrast is scored inside a 2 s window, so it needs word timings for
the hypotheses and both references. By default the code uses each model's own
timestamps. The paper instead applies one NeMo CTC forced alignment to all three
models, which keeps the windowed contrast from depending on each model's own
timestamp quality. That path has three steps:

```bash
# 1. align every model's hypotheses with the same CTC model
python tools/align_outputs.py \
  --run outputs/primary/temporal_history/runs/whisper.jsonl \
  --output outputs/primary/common_alignment/whisper.jsonl \
  --model-path /path/to/parakeet-ctc-1.1b.nemo \
  --model-revision 20e63a0fed6aedba145b74b826dbd41df0941730
# repeat for qwen3 and ctc, then

# 2. point the scoring stage at those manifests
python tools/make_alignment_scoring_config.py \
  --base-config configs/primary.local.yaml \
  --alignment-root outputs/primary/common_alignment \
  --output configs/primary.scoring.local.yaml

# 3. score and analyze with the alignment-aware config
source-capture history score   --config configs/primary.scoring.local.yaml --model whisper
source-capture history analyze --config configs/primary.scoring.local.yaml
```

`tools/compare_alignment_sensitivity.py` reruns the contrast under both native
and common alignment and reports whether the directions agree. If common
alignment is required and no manifest is configured, the scoring stage stops
with an error rather than quietly dropping every timestamp and leaving the
windowed analysis empty.

## Tools

`tools/` holds the pipeline steps that sit outside the package:

| Tool | Purpose |
|---|---|
| `align_outputs.py` | force-align a run file with the shared NeMo CTC model |
| `make_alignment_scoring_config.py` | write a scoring config that points all three models at their alignment manifests |
| `compare_alignment_sensitivity.py` | check the history contrast under native versus common alignment |
| `validate_complete_run.py` | verify a run covers its manifest exactly once with zero inference errors |
| `merge_sharded_runs.py` | merge deterministic inference shards into one canonical run file |
| `qwen3_asr_service.py` | local HTTP service for Qwen3-ASR, the endpoint the `qwen_http` backend talks to |

## Dataset specifications and reported values

- `protocol/` records the fixed evaluation inputs used by the paper: tracks,
  vocal crop offsets, speech utterances, and replication selections.
- `results/reported_effects.json` contains the Position and History values used
  in Fig. 2 and can be used to check an independently generated summary.

## Known gaps

- **Primary word-level lyric annotations.** The primary evaluation uses
  one `<Track Name>_align.csv` per track, with onset in seconds and word.
  The converter `source_capture.preprocessing.musdb_lyrics` turns this input
  into the manifest consumed by dataset preparation. The upstream MUSDB18
  lyrics release provides line-level annotations; obtain the word-level CSV
  bundle separately when reproducing the primary set. The replication set
  instead generates word timings from the published line-level annotations
  with `source_capture.preprocessing.replication_lyrics`.
- **The DeepFilterNet binary.** The front-end experiment shells out to it. The
  reported run used `deep-filter-0.5.6-x86_64-unknown-linux-musl` from the
  DeepFilterNet release page; the binary is not redistributed here.
- **Model revisions.** Nothing in this repository downloads model weights.
  Record the revision hash of every checkpoint you use alongside your local
  configuration. The Parakeet revision used for common alignment is recorded
  in the primary dataset manifest.
- **Run files and audio are not distributed.** Only the compact reference
  values in `results/` are included.

## Inputs and outputs

The code expects MUSDB18-HQ, the MUSDB18 lyrics extension, and LibriSpeech
`test-clean`, obtained from their official distributions. Generated mixtures,
ASR hypotheses, and summaries are written below the configured output path.

Dataset conversion utilities are available under
`source_capture.preprocessing`: `musdb_audio`, `musdb_lyrics`, `librispeech`,
and `replication_lyrics`. Each can be run with `python -m ... --help`;
`librispeech` additionally needs `pyarrow`.

To convert separately obtained primary word-onset annotations, place one
`<Track Name>_align.csv` file per track in an alignment directory. Each row
must contain the onset time in seconds followed by the lyric word. The files
may be UTF-8 (with or without a BOM) or Windows-1252. Convert them to the
manifest consumed by dataset preparation with:

```bash
PYTHONPATH=src python -m source_capture.preprocessing.musdb_lyrics \
  --alignments /path/to/primary_word_alignments \
  --tracks-root /path/to/MUSDB18-HQ \
  --output /path/to/labels/musdb18_word_onsets.jsonl
```

The alignment CSVs are not redistributed here because they are derived from
the MUSDB18 material and follow its upstream access terms. The repository
contains the conversion code and the exact primary track/crop specification;
the same arrangement is used for the replication labels, which are generated
from the public line-level lyrics extension by
`source_capture.preprocessing.replication_lyrics`.

Each stage writes a JSONL manifest below the configured output directory.
Every row retains the pair identifier, condition, source references, model
output, and source-exclusive scores needed by the next stage.

The isolated-source validation runs as two commands on the prepared pair
manifest, with the same model configuration as the main experiments:

```bash
source-capture-validate transcribe --config configs/primary.local.yaml \
  --pairs outputs/primary/shared/pairs.jsonl \
  --output outputs/primary/single_source/whisper.jsonl --model whisper
source-capture-validate analyze \
  --output outputs/primary/single_source/whisper.jsonl \
  --summary outputs/primary/single_source/whisper.summary.json \
  --config configs/primary.local.yaml --model whisper --pairs outputs/primary/shared/pairs.jsonl
```

## License and citation

MIT, see `LICENSE`. Citation metadata is in `CITATION.cff`.
