from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .core.audio import (
    active_fraction,
    read_audio,
    required_windows_have_energy,
    rms,
    select_active_crops,
    vad_onset,
    write_pcm16,
)
from .config import output_root
from .core.io import load_json_or_jsonl, sha256_file, stable_id, write_jsonl


AUDIO_SUFFIXES = {".flac", ".wav"}


def prepare_pairs(config: dict[str, Any]) -> Path:
    root = output_root(config)
    shared = root / "shared"
    speech_dir = shared / "speech"
    vocal_dir = shared / "vocals"
    speech_dir.mkdir(parents=True, exist_ok=True)
    vocal_dir.mkdir(parents=True, exist_ok=True)
    sample_rate = int(config["audio"]["sample_rate"])
    duration_s = float(config["audio"]["clip_duration_s"])
    width = round(sample_rate * duration_s)
    seed = int(config["project"]["seed"])
    selection = config["selection"]

    speech_alignments = _load_speech_alignments(config)
    speech_rows, speech_checks = _prepare_speech(
        config, speech_dir, sample_rate, width, seed, speech_alignments
    )
    lyric_rows = _load_lyrics(config)
    vocal_rows, vocal_checks = _prepare_vocals(
        config, vocal_dir, sample_rate, duration_s, lyric_rows, seed
    )

    pairs: list[dict[str, Any]] = []
    for vocal in vocal_rows:
        for speech in speech_rows:
            pair_id = stable_id(speech["speech_id"], vocal["vocal_crop_id"])
            pairs.append(
                {
                    "pair_id": pair_id,
                    "split": vocal["split"],
                    **speech,
                    **vocal,
                }
            )
    pairs_path = shared / "pairs.jsonl"
    write_jsonl(pairs_path, pairs)
    checks = {
        "speech": speech_checks,
        "vocals": vocal_checks,
        "counts": {
            "speech": len(speech_rows),
            "vocal_crops": len(vocal_rows),
            "pairs": len(pairs),
            "development_pairs": sum(row["split"] == "development" for row in pairs),
            "evaluation_pairs": sum(row["split"] == "evaluation" for row in pairs),
        },
    }
    (shared / "preparation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return pairs_path


def _prepare_speech(
    config: dict[str, Any],
    destination: Path,
    sample_rate: int,
    width: int,
    seed: int,
    speech_alignments: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = Path(config["datasets"]["librispeech_test_clean"])
    transcripts = _librispeech_transcripts(dataset)
    excluded = set(config["selection"].get("excluded_speech_ids", []))
    excluded_speakers = {
        str(value)
        for value in config["selection"].get("excluded_speech_speaker_ids", [])
    }
    paths = sorted(
        path for path in dataset.rglob("*") if path.suffix.lower() in AUDIO_SUFFIXES
    )
    rng = random.Random(seed)
    rng.shuffle(paths)
    target_count = int(config["selection"]["n_speech_utterances"])
    vad_config = config["audio"]["vad"]
    selected: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    selected_speakers: set[str] = set()

    for path in paths:
        speech_id = path.stem
        if speech_id in excluded or speech_id not in transcripts:
            continue
        speaker_id = speech_id.split("-")[0]
        if speaker_id in excluded_speakers:
            continue
        try:
            audio = read_audio(path, sample_rate)
            onset, vad_stats = vad_onset(
                audio,
                sample_rate,
                frame_ms=float(vad_config["frame_ms"]),
                hop_ms=float(vad_config["hop_ms"]),
                threshold_db_below_peak=float(vad_config["threshold_db_below_peak"]),
                min_consecutive_frames=int(vad_config["min_consecutive_frames"]),
                pre_roll_ms=float(vad_config["pre_roll_ms"]),
            )
            crop = audio[onset : onset + width]
            if len(crop) < width:
                continue
            windows_ok, window_checks = required_windows_have_energy(
                crop,
                sample_rate,
                config["selection"].get("required_activity_windows_s", []),
                float(
                    config["selection"].get(
                        "component_window_min_db_below_clip_rms", 20.0
                    )
                ),
            )
            if not windows_ok:
                checks.append(
                    {
                        "source": str(path),
                        "speech_id": speech_id,
                        "status": "rejected",
                        "reason": "target speech is inactive in a required History window",
                        "required_window_checks": window_checks,
                    }
                )
                continue
            stats = {
                "source": str(path),
                "speech_id": speech_id,
                "speaker_id": speaker_id,
                "crop_start_s": onset / sample_rate,
                "crop_rms": rms(crop),
                **vad_stats,
            }
            item = (path, stats, crop)
            if speaker_id in selected_speakers:
                continue
            if config["scoring"].get("require_speech_alignment", True) and not speech_alignments.get(
                speech_id
            ):
                checks.append(
                    {
                        "source": str(path),
                        "speech_id": speech_id,
                        "status": "rejected",
                        "reason": "missing required speech word alignment",
                    }
                )
                continue
            _save_speech(
                item,
                destination,
                sample_rate,
                transcripts,
                speech_alignments,
                selected,
                checks,
            )
            selected_speakers.add(speaker_id)
            if len(selected) == target_count:
                break
        except (ValueError, RuntimeError) as exc:
            checks.append({"source": str(path), "status": "rejected", "reason": str(exc)})

    if len(selected) < target_count:
        raise RuntimeError(
            f"only found {len(selected)} distinct-speaker speech utterances with 10 s "
            f"after VAD onset and the required alignment/activity checks; "
            f"requested {target_count}"
        )
    return selected, checks


def _save_speech(
    item: tuple[Path, dict[str, Any], Any],
    destination: Path,
    sample_rate: int,
    transcripts: dict[str, str],
    speech_alignments: dict[str, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> None:
    _, stats, crop = item
    output = destination / f"{stats['speech_id']}.wav"
    write_pcm16(output, crop, sample_rate)
    shifted_words = _shift_words(
        speech_alignments.get(stats["speech_id"]),
        offset_s=float(stats["crop_start_s"]),
        window_s=len(crop) / sample_rate,
    )
    speech_reference = (
        " ".join(item["word"] for item in shifted_words)
        if shifted_words
        else transcripts[stats["speech_id"]]
    )
    selected.append(
        {
            "speech_id": stats["speech_id"],
            "speech_speaker_id": stats["speaker_id"],
            "speech_path": str(output.resolve()),
            "speech_sha256": sha256_file(output),
            "speech_reference": speech_reference,
            "speech_words": shifted_words,
            "speech_source_path": stats["source"],
            "speech_crop_start_s": stats["crop_start_s"],
        }
    )
    checks.append({**stats, "status": "selected", "output": str(output.resolve())})


def _prepare_vocals(
    config: dict[str, Any],
    destination: Path,
    sample_rate: int,
    duration_s: float,
    lyrics: dict[str, list[dict[str, Any]]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = Path(config["datasets"]["musdb18hq_root"])
    paths = sorted(dataset.rglob("vocals.wav"))
    excluded = set(config["selection"].get("excluded_track_ids", []))
    required = [
        str(value) for value in config["selection"].get("required_track_ids", [])
    ]
    if len(required) != len(set(required)):
        raise ValueError("selection.required_track_ids contains duplicates")
    track_paths: list[tuple[str, Path]] = []
    for path in paths:
        track_id = path.parent.relative_to(dataset).as_posix()
        if track_id not in excluded:
            track_paths.append((track_id, path))
    requested_tracks = int(config["selection"]["n_tracks"])
    track_paths = _order_track_paths(track_paths, required, requested_tracks, seed)
    accepted_by_track: list[tuple[str, Path, list[tuple[int, Any, float]]]] = []
    checks: list[dict[str, Any]] = []
    crops_per_track = int(config["selection"]["crops_per_track"])

    for track_id, path in track_paths:
        try:
            audio = read_audio(path, sample_rate)
            require_crop_lyrics = config["scoring"].get(
                "require_crop_aligned_lyrics", True
            )
            require_words = config["scoring"].get(
                "require_word_aligned_references_for_temporal_history", True
            )

            def valid_reference(start: int, _: Any) -> bool:
                if not require_crop_lyrics:
                    return True
                reference, words, scope = _lyrics_for_crop(
                    lyrics.get(track_id, []),
                    start / sample_rate,
                    start / sample_rate + duration_s,
                )
                return bool(
                    scope == "crop" and reference and (words or not require_words)
                )

            crops = select_active_crops(
                audio,
                sample_rate,
                duration_s=duration_s,
                count=crops_per_track,
                hop_s=float(config["selection"]["vocal_crop_hop_s"]),
                min_active_fraction=float(config["selection"]["vocal_min_active_fraction"]),
                threshold_db_below_peak=float(
                    config["selection"]["vocal_activity_threshold_db_below_track_peak"]
                ),
                required_activity_windows_s=config["selection"].get(
                    "required_activity_windows_s", []
                ),
                component_window_min_db_below_clip_rms=float(
                    config["selection"].get(
                        "component_window_min_db_below_clip_rms", 20.0
                    )
                ),
                candidate_validator=valid_reference,
            )
            if len(crops) < crops_per_track:
                checks.append(
                    {
                        "track_id": track_id,
                        "source": str(path),
                        "status": "rejected",
                        "reason": f"only {len(crops)} active crops; need {crops_per_track}",
                    }
                )
                continue
            if require_crop_lyrics:
                lyric_checks = [
                    _lyrics_for_crop(
                        lyrics.get(track_id, []),
                        start / sample_rate,
                        start / sample_rate + duration_s,
                    )
                    for start, _, _ in crops
                ]
                if any(
                    scope != "crop" or not reference or (require_words and not words)
                    for reference, words, scope in lyric_checks
                ):
                    checks.append(
                        {
                            "track_id": track_id,
                            "source": str(path),
                            "status": "rejected",
                            "reason": "selected crops lack required crop-aligned lyric text/words",
                        }
                    )
                    continue
            accepted_by_track.append((track_id, path, crops))
            if len(accepted_by_track) == requested_tracks:
                break
        except (ValueError, RuntimeError) as exc:
            checks.append(
                {"track_id": track_id, "source": str(path), "status": "rejected", "reason": str(exc)}
            )
    if len(accepted_by_track) < requested_tracks:
        rejection_details = "; ".join(
            f"{row.get('track_id', '<unknown>')}: {row.get('reason', '<unknown>')}"
            for row in checks
            if row.get("status") == "rejected"
        )
        raise RuntimeError(
            f"only found {len(accepted_by_track)} tracks with {crops_per_track} active crops; "
            f"requested {requested_tracks}. Rejections: {rejection_details}"
        )
    split_count = round(
        len(accepted_by_track) * float(config["selection"]["split_development_fraction"])
    )
    development_tracks = {track_id for track_id, _, _ in accepted_by_track[:split_count]}
    selected: list[dict[str, Any]] = []
    for track_id, path, crops in sorted(accepted_by_track):
        try:
            for crop_index, (start, crop, fraction) in enumerate(crops):
                crop_start_s = start / sample_rate
                lyric_reference, lyric_words, lyric_scope = _lyrics_for_crop(
                    lyrics.get(track_id, []), crop_start_s, crop_start_s + duration_s
                )
                crop_id = stable_id(track_id, crop_start_s)
                output = destination / f"{crop_id}.wav"
                write_pcm16(output, crop, sample_rate)
                selected.append(
                    {
                        "track_id": track_id,
                        "vocal_crop_id": crop_id,
                        "vocal_crop_index": crop_index,
                        "vocal_path": str(output.resolve()),
                        "vocal_sha256": sha256_file(output),
                        "vocal_source_path": str(path),
                        "vocal_crop_start_s": crop_start_s,
                        "vocal_active_fraction": fraction,
                        "lyric_reference": lyric_reference,
                        "lyric_words": lyric_words,
                        "lyric_reference_scope": lyric_scope,
                        "split": (
                            "development" if track_id in development_tracks else "evaluation"
                        ),
                    }
                )
                checks.append(
                    {
                        "track_id": track_id,
                        "crop_start_s": crop_start_s,
                        "active_fraction": fraction,
                        "crop_rms": rms(crop),
                        "has_lyrics": bool(lyric_reference),
                        "status": "selected",
                        "output": str(output.resolve()),
                    }
                )
        except (ValueError, RuntimeError) as exc:
            checks.append(
                {"track_id": track_id, "source": str(path), "status": "rejected", "reason": str(exc)}
            )
    if not selected:
        raise RuntimeError("no vocal crops were selected")
    if not config["datasets"].get("allow_missing_lyrics_during_generation", False):
        missing = [row["vocal_crop_id"] for row in selected if not row["lyric_reference"]]
        if missing:
            raise RuntimeError(f"{len(missing)} selected vocal crops have no lyric reference")
    return selected, checks


def _order_track_paths(
    track_paths: list[tuple[str, Path]],
    required: list[str],
    requested_tracks: int,
    seed: int,
) -> list[tuple[str, Path]]:
    if required:
        if requested_tracks != len(required):
            raise ValueError(
                "selection.n_tracks must equal the number of required_track_ids"
            )
        available = dict(track_paths)
        missing = [track_id for track_id in required if track_id not in available]
        if missing:
            raise RuntimeError(
                "required tracks are missing or excluded: " + ", ".join(missing)
            )
        return [(track_id, available[track_id]) for track_id in required]
    output = list(track_paths)
    random.Random(seed + 1).shuffle(output)
    return output


def _librispeech_transcripts(root: Path) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for path in root.rglob("*.trans.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utterance_id, text = line.split(maxsplit=1)
            transcripts[utterance_id] = text
    if not transcripts:
        raise RuntimeError(f"no LibriSpeech transcript files found below {root}")
    return transcripts


def _load_lyrics(config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    path = config["datasets"].get("lyrics_manifest")
    if not path or str(path).startswith("__SET_ME_"):
        return {}
    rows = load_json_or_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "track_id" not in row or "lyrics" not in row:
            raise ValueError("each lyrics record needs track_id and lyrics")
        grouped[str(row["track_id"])].append(row)
    return grouped


def _load_speech_alignments(
    config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    path = config["datasets"].get("speech_alignment_manifest")
    if not path or str(path).startswith("__SET_ME_"):
        return {}
    rows = load_json_or_jsonl(path)
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "speech_id" not in row or "words" not in row:
            raise ValueError("each speech alignment record needs speech_id and words")
        output[str(row["speech_id"])] = list(row["words"])
    return output


def _lyrics_for_crop(
    rows: list[dict[str, Any]], crop_start_s: float, crop_end_s: float
) -> tuple[str, list[dict[str, Any]] | None, str | None]:
    for row in rows:
        start = row.get("crop_start_s")
        end = row.get("crop_end_s")
        if start is None and end is None:
            words = _shift_words(row.get("words"), offset_s=crop_start_s, window_s=crop_end_s-crop_start_s)
            reference = (
                " ".join(item["word"] for item in words)
                if words
                else str(row.get("lyrics", ""))
            )
            return reference, words, "track"
        if start is not None and end is not None:
            overlap = min(crop_end_s, float(end)) - max(crop_start_s, float(start))
            if overlap >= 0.8 * (crop_end_s - crop_start_s):
                words = _shift_words(row.get("words"), offset_s=crop_start_s, window_s=crop_end_s-crop_start_s)
                reference = (
                    " ".join(item["word"] for item in words)
                    if words
                    else str(row.get("lyrics", ""))
                )
                return reference, words, "crop"
    return "", None, None


def _shift_words(
    words: list[dict[str, Any]] | None,
    *,
    offset_s: float,
    window_s: float | None,
) -> list[dict[str, Any]] | None:
    if not words:
        return None
    shifted: list[dict[str, Any]] = []
    for item in words:
        start = float(item["start"]) - offset_s
        end = float(item["end"]) - offset_s
        if end <= 0 or (window_s is not None and start >= window_s):
            continue
        shifted.append(
            {
                "word": str(item["word"]),
                "start": max(0.0, start),
                "end": end if window_s is None else min(window_s, end),
            }
        )
    return shifted or None
