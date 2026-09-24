#!/usr/bin/env python3
"""Produce word-level lyric alignments for MUSDB18 train tracks.

For every track, each annotated lyrics section (Schulze-Forster MUSDB18
lyrics extension, ``MM:SS MM:SS <prop> <text>`` lines) is force-aligned
independently with the specified NeMo CTC aligner (nvidia/parakeet-ctc-1.1b,
Viterbi), and word timings are merged into a full-track reference in the
same JSONL schema as data/labels/musdb18_lyrics.jsonl.

Sections marked with property "d" (no singing) or with empty text are
skipped. A per-track validation report is printed so alignment failures can
be caught before replication inference begins.

Example:
    python -m source_capture.preprocessing.replication_lyrics \
        --stems-root /path/to/musdb16k_train \
        --lyrics-dir /path/to/musdb18_train_lyrics \
        --output /path/to/musdb18_train_words.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..core.audio import read_pcm16, write_pcm16
from ..core.models import NeMoCTCBackend
from ..core.alignment import _extract_aligned_words

MODEL_CONFIG = {
    "backend": "nemo_ctc",
    "model_name": "nvidia/parakeet-ctc-1.1b",
    "device": "cuda",
    "batch_size": 16,
    "word_timestamps": True,
    "decoding_strategy": "greedy_batch",
    "external_language_model": None,
}

PAD_S = 0.10  # audio padding around each section before alignment


def _parse_section(line: str) -> tuple[float, float, str, str] | None:
    parts = line.strip().split(maxsplit=3)
    if len(parts) < 4:
        return None

    def _to_seconds(value: str) -> float:
        minutes, seconds = value.split(":")
        return int(minutes) * 60.0 + float(seconds)

    try:
        start, end = _to_seconds(parts[0]), _to_seconds(parts[1])
    except (ValueError, IndexError):
        return None
    prop, text = parts[2], parts[3].strip()
    return start, end, prop, text


def _align_sections(
    backend: NeMoCTCBackend,
    items: list[dict],
) -> dict[int, dict]:
    """Align section slices; returns {section_index: {"words": [...], "error": ...}}."""
    from nemo.collections.asr.parts.utils.aligner_utils import (
        add_t_start_end_to_utt_obj,
        get_batch_variables,
        viterbi_decoding,
    )

    output: dict[int, dict] = {}
    batch = 16
    for offset in range(0, len(items), batch):
        chunk = items[offset : offset + batch]
        variables = get_batch_variables(
            audio=[item["slice_path"] for item in chunk],
            model=backend.model,
            segment_separators=[],
            word_separator=" ",
            align_using_pred_text=False,
            gt_text_batch=[item["text"] for item in chunk],
            output_timestep_duration=backend.time_stride,
            verbose=False,
        )
        log_probs, targets, lengths, target_lengths, utterances, timestep = variables
        alignments = viterbi_decoding(
            log_probs, targets, lengths, target_lengths,
            viterbi_device=next(backend.model.parameters()).device,
        )
        for item, utterance, alignment in zip(chunk, utterances, alignments):
            timed = add_t_start_end_to_utt_obj(utterance, alignment, timestep)
            words = [
                {
                    "word": word["word"],
                    "start": round(word["start"] + item["slice_start"], 6),
                    "end": round(word["end"] + item["slice_start"], 6),
                }
                for word in _extract_aligned_words(timed)
            ]
            expected = len(item["text"].split())
            error = None
            if len(words) != expected:
                error = f"word count mismatch: aligned {len(words)} vs expected {expected}"
            output[item["index"]] = {"words": words, "error": error}
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stems-root", required=True)
    parser.add_argument("--lyrics-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tracks", nargs="*", default=None, help="optional subset of track names")
    args = parser.parse_args()

    stems_root = Path(args.stems_root)
    lyrics_dir = Path(args.lyrics_dir)
    backend = NeMoCTCBackend(MODEL_CONFIG)
    report: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp, Path(args.output).open("w", encoding="utf-8") as out:
        tmpdir = Path(tmp)
        lyrics_files = sorted(lyrics_dir.glob("*.txt"))
        if args.tracks:
            wanted = set(args.tracks)
            lyrics_files = [path for path in lyrics_files if path.stem in wanted]
        for file_index, lyrics_path in enumerate(lyrics_files, 1):
            track = lyrics_path.stem
            vocal_path = stems_root / track / "vocals.wav"
            if not vocal_path.exists():
                print(f"SKIP {track}: no vocals.wav", flush=True)
                continue
            audio, sample_rate = read_pcm16(vocal_path)
            duration = len(audio) / sample_rate
            sections = []
            for line in lyrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
                parsed = _parse_section(line)
                if parsed is None:
                    continue
                start, end, prop, text = parsed
                if prop == "d" or not text:
                    continue
                sections.append((start, end, text))
            items = []
            for index, (start, end, text) in enumerate(sections):
                slice_start = max(0.0, start - PAD_S)
                lo = round(slice_start * sample_rate)
                hi = min(len(audio), round((end + PAD_S) * sample_rate))
                slice_path = tmpdir / f"{file_index:03d}_{index:03d}.wav"
                write_pcm16(slice_path, audio[lo:hi], sample_rate)
                items.append(
                    {"index": index, "text": text, "slice_path": str(slice_path), "slice_start": slice_start}
                )
            try:
                aligned = _align_sections(backend, items)
            except Exception as exc:  # fall back to per-section alignment
                aligned = {}
                for item in items:
                    try:
                        aligned.update(_align_sections(backend, [item]))
                    except Exception as inner:
                        aligned[item["index"]] = {"words": [], "error": f"{type(inner).__name__}: {inner}"}
            words: list[dict] = []
            failures = 0
            out_of_bounds = 0
            for index, (start, end, _) in enumerate(sections):
                result = aligned.get(index, {"words": [], "error": "missing"})
                if result["error"] or not result["words"]:
                    failures += 1
                for word in result["words"]:
                    midpoint = (word["start"] + word["end"]) / 2.0
                    if midpoint < start - 0.75 or midpoint > end + 0.75:
                        out_of_bounds += 1
                    words.append(word)
            words.sort(key=lambda item: item["start"])
            full_text = " ".join(text for _, _, text in sections)
            out.write(
                json.dumps(
                    {
                        "track_id": track,
                        "crop_start_s": 0.0,
                        "crop_end_s": round(duration, 6),
                        "lyrics": full_text,
                        "words": words,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            report.append(
                {
                    "track_id": track,
                    "duration_s": round(duration, 3),
                    "sections": len(sections),
                    "section_failures": failures,
                    "words": len(words),
                    "words_out_of_bounds": out_of_bounds,
                }
            )
            print(
                f"[{file_index}/{len(lyrics_files)}] {track}: "
                f"{len(words)} words, {failures}/{len(sections)} section failures, "
                f"{out_of_bounds} out-of-bounds",
                flush=True,
            )
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
