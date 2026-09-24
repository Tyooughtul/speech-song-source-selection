#!/usr/bin/env python3
"""Resolve per-model common alignment manifests into a scoring-only config.

The windowed history analysis in the paper uses one NeMo CTC alignment applied
identically to all three models. This writes a copy of the configuration with
each model pointed at its alignment manifest and the external-alignment
requirement switched on, which is the setting the scoring stage needs.

Example:
    python tools/make_alignment_scoring_config.py \
        --base-config configs/primary.local.yaml \
        --alignment-root outputs/primary/common_alignment \
        --output configs/primary.scoring.local.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

FILENAMES = {"whisper": "whisper.jsonl", "qwen3": "qwen3.jsonl", "ctc": "ctc.jsonl"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--alignment-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.base_config).read_text(encoding="utf-8"))
    root = Path(args.alignment_root).resolve()
    for model, filename in FILENAMES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(path)
        config["models"][model]["word_alignment_manifest"] = str(path)
        config["models"][model]["prefer_external_word_alignment"] = True
        config["models"][model]["require_external_word_alignment"] = True

    output = Path(args.output)
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
