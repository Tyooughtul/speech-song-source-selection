"""Validate that each ASR model recognizes both isolated sources."""

from __future__ import annotations

import argparse

from .config import load_config
from .validation import run_single_source, write_single_source_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("transcribe", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", required=True, choices=("whisper", "qwen3", "ctc"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "transcribe":
        print(run_single_source(load_config(args.config), args.pairs, args.output, args.model, overwrite=args.overwrite))
    else:
        if not args.summary:
            parser.error("--summary is required for analyze")
        print(write_single_source_summary(args.output, args.summary))
    return 0
