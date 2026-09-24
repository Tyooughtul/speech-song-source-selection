#!/usr/bin/env python3
"""Validate and merge deterministic ASR run shards into one canonical JSONL.

Long runs were sharded across processes. This refuses to merge unless the shards
together cover the manifest exactly once with zero inference errors, and writes
rows in manifest order so the canonical file is reproducible.

Example:
    python tools/merge_sharded_runs.py \
        --manifest outputs/primary/onset_position/manifest.jsonl \
        --shard outputs/primary/shards/whisper.0.jsonl \
        --shard outputs/primary/shards/whisper.1.jsonl \
        --output outputs/primary/onset_position/runs/whisper.jsonl
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", action="append", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    manifest_ids = [str(row["sample_id"]) for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise RuntimeError("canonical manifest contains duplicate sample IDs")

    by_id: dict[str, dict[str, Any]] = {}
    for value in args.shard:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            sample_id = str(row["sample_id"])
            if sample_id in by_id:
                raise RuntimeError(f"duplicate run sample ID across shards: {sample_id}")
            by_id[sample_id] = row

    missing = sorted(set(manifest_ids) - set(by_id))
    unexpected = sorted(set(by_id) - set(manifest_ids))
    errors = [sample_id for sample_id, row in by_id.items() if row.get("error")]
    if missing or unexpected or errors:
        raise RuntimeError(
            "refusing canonical merge: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, errors={len(errors)}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in manifest_ids:
            handle.write(json.dumps(by_id[sample_id], ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(f"{output}: {len(manifest_ids)} unique rows, zero inference errors")


if __name__ == "__main__":
    main()
