#!/usr/bin/env python3
"""Check that a run file is complete and free of row-level inference failures.

Checks the invariant required before scoring: every
manifest row appears exactly once, and no row carries an inference error. It
reads only identifiers and error fields, so it can run before any condition
summary is opened.

Example:
    python tools/validate_complete_run.py \
        --manifest outputs/primary/temporal_history/manifest.jsonl \
        --run outputs/primary/temporal_history/runs/whisper.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    run = read_jsonl(Path(args.run))
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    run_ids = [str(row["sample_id"]) for row in run]
    errors = [str(row["sample_id"]) for row in run if row.get("error")]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise RuntimeError("manifest contains duplicate sample IDs")
    if len(run_ids) != len(set(run_ids)):
        raise RuntimeError("run contains duplicate sample IDs")
    missing = set(manifest_ids) - set(run_ids)
    unexpected = set(run_ids) - set(manifest_ids)
    if missing or unexpected or errors:
        raise RuntimeError(
            f"incomplete run: missing={len(missing)}, unexpected={len(unexpected)}, "
            f"errors={len(errors)}"
        )
    print(f"{args.run}: {len(run)} unique rows, zero inference errors")


if __name__ == "__main__":
    main()
