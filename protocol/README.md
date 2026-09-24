# Dataset specifications

The JSON files in this directory describe the fixed evaluation inputs used by
the paper. They are data-selection manifests, not records of the authors'
development process or compute environment.

- `primary_evaluation_set.json` — the primary evaluation tracks, vocal crop offsets, speech
  utterances, and audio digests.
- `replication_set.json` — the independent track and speaker selection and
  corresponding experiment settings.

The source audio, model weights, run files, and private annotations are not
distributed here. Use the conversion utilities in `source_capture.preprocessing`
to construct the public input manifests from the upstream datasets and any
separately obtained word-level lyric annotations.

`results/reported_effects.json` contains the Position and History values plotted
in Fig. 2 for checking an independent reproduction.
