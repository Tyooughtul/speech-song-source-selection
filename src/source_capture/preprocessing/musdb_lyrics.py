#!/usr/bin/env python3
"""Convert MUSDB word-onset CSV files to the Position-History lyrics manifest format."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import soundfile as sf


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _track_names(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for vocal in sorted(root.rglob("vocals.wav")):
        name = vocal.parent.name
        normalized = _key(name)
        if normalized in output and output[normalized] != name:
            raise ValueError(f"ambiguous normalized track name: {name}")
        output[normalized] = name
    if not output:
        raise RuntimeError(f"no prepared vocals.wav files found below {root}")
    return output


def _read_onsets(path: Path) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    raw = path.read_bytes()
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # The published alignment bundle mixes UTF-8 and Windows-1252 CSVs.
        # Keep decoding strict so malformed input is never silently discarded.
        decoded = raw.decode("cp1252")
    for row in csv.reader(decoded.splitlines()):
        if len(row) < 2:
            continue
        word = row[1].strip()
        if not word:
            continue
        rows.append((float(row[0]), word))
    return sorted(rows)


def build(alignments: Path, tracks_root: Path, output: Path) -> None:
    tracks = _track_names(tracks_root)
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for path in sorted(alignments.glob("*_align.csv")):
        source_name = path.stem.removesuffix("_align")
        track_id = tracks.get(_key(source_name))
        if track_id is None:
            missing.append(source_name)
            continue
        onsets = _read_onsets(path)
        if not onsets:
            continue
        words = []
        for index, (start, word) in enumerate(onsets):
            next_start = onsets[index + 1][0] if index + 1 < len(onsets) else start + 0.8
            end = min(next_start, start + 1.5)
            words.append({"word": word, "start": start, "end": max(start + 0.05, end)})
        vocal_path = tracks_root / track_id / "vocals.wav"
        info = sf.info(vocal_path)
        records.append(
            {
                "track_id": track_id,
                "crop_start_s": 0.0,
                "crop_end_s": float(info.frames / info.samplerate),
                "lyrics": " ".join(word for _, word in onsets),
                "words": words,
            }
        )
    if not records:
        raise RuntimeError("no alignment CSV matched a prepared MUSDB track")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(records)} tracks to {output}")
    if missing:
        print(f"unmatched alignment files ({len(missing)}): {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.alignments.expanduser().resolve(),
        args.tracks_root.expanduser().resolve(),
        args.output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
