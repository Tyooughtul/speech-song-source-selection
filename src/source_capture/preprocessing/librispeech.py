#!/usr/bin/env python3
"""Convert the LibriSpeech MFA parquet split to source-capture JSONL.

Requires pyarrow, which is not part of the core install:
    python -m pip install 'pyarrow>=14'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(source: Path, output: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "this converter needs pyarrow: python -m pip install 'pyarrow>=14'"
        ) from exc
    table = pq.read_table(source, columns=["id", "words"])
    records = table.to_pylist()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in records:
            speech_id = str(row["id"])
            words = []
            for item in row.get("words") or []:
                word = str(item.get("word", "")).strip()
                if not word:
                    continue
                words.append(
                    {
                        "word": word,
                        "start": float(item["start"]),
                        "end": float(item["end"]),
                    }
                )
            if not words:
                continue
            handle.write(
                json.dumps(
                    {"speech_id": speech_id, "words": words},
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    print(f"wrote {written} utterance alignments to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.input.expanduser().resolve(), args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
