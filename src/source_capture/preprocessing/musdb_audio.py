#!/usr/bin/env python3
"""Decode MUSDB18 train stems (mp3 tar) into 16 kHz mono wavs.

Produces the same on-disk layout expected by the primary-set preparation
pipeline can consume it unchanged:

    <output_root>/<Track Name>/vocals.wav   (16 kHz mono PCM16)
    <output_root>/<Track Name>/accomp.wav   (drums + bass + other, 16 kHz mono)

Only tracks with a lyrics-extension file are decoded. Prints one JSON line
per track with duration and SHA-256 digests for reproducibility checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 16000


def _decode(source: Path, destination: Path, extra_filter: str | None = None) -> None:
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(source)]
    if extra_filter:
        command += ["-af", extra_filter]
    command += ["-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(destination)]
    subprocess.run(command, check=True)


def _decode_mix(sources: list[Path], destination: Path) -> None:
    command = ["ffmpeg", "-y", "-v", "error"]
    for source in sources:
        command += ["-i", str(source)]
    command += [
        "-filter_complex",
        f"amix=inputs={len(sources)}:normalize=0",
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(destination),
    ]
    subprocess.run(command, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", required=True, help="musdb18hq train tar with mp3 stems")
    parser.add_argument("--lyrics-dir", required=True, help="directory of lyrics-extension .txt files")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    lyrics_dir = Path(args.lyrics_dir)
    output_root = Path(args.output_root)
    track_names = sorted(path.stem for path in lyrics_dir.glob("*.txt"))
    print(f"{len(track_names)} tracks with lyrics files", flush=True)

    with tarfile.open(args.tar) as archive, tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for index, track in enumerate(track_names, 1):
            stem_names = {
                part: f"{track}.{part}.mp3"
                for part in ("vocals", "drums", "bass", "other")
            }
            missing = [name for name in stem_names.values() if name not in members]
            if missing:
                print(f"SKIP {track}: missing tar members {missing}", flush=True)
                continue
            extracted: dict[str, Path] = {}
            for part, name in stem_names.items():
                target = tmpdir / name
                with archive.extractfile(members[name]) as source, target.open("wb") as out:
                    out.write(source.read())
                extracted[part] = target
            track_dir = output_root / track
            track_dir.mkdir(parents=True, exist_ok=True)
            vocals = track_dir / "vocals.wav"
            accomp = track_dir / "accomp.wav"
            _decode(extracted["vocals"], vocals)
            _decode_mix([extracted["drums"], extracted["bass"], extracted["other"]], accomp)
            print(
                json.dumps(
                    {
                        "track_id": track,
                        "duration_s": round(_duration_s(vocals), 3),
                        "vocals_sha256": _sha256(vocals),
                        "accomp_sha256": _sha256(accomp),
                    }
                ),
                flush=True,
            )
            print(f"[{index}/{len(track_names)}] {track}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
