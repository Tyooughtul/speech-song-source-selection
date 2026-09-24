from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

from .core.models import create_backend
from .core.io import append_jsonl, read_jsonl, stable_id
from .core.scoring import attribute_tokens, normalize_words


def run_single_source(
    config: dict[str, Any],
    pairs_path: str | Path,
    output_path: str | Path,
    model_name: str,
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(output_path)
    if destination.exists() and overwrite:
        destination.unlink()
    completed = (
        {str(row["sample_id"]) for row in read_jsonl(destination)}
        if destination.exists()
        else set()
    )
    items = _unique_sources(read_jsonl(pairs_path))
    items = [row for row in items if row["sample_id"] not in completed]
    backend_config = config["models"][model_name]
    backend = create_backend(model_name, backend_config)
    chunk_size = int(backend_config.get("transcribe_chunk_size", 256))
    started = time.time()
    written = 0
    for offset in range(0, len(items), chunk_size):
        chunk = items[offset : offset + chunk_size]
        try:
            predictions = backend.transcribe_batch([row["audio_path"] for row in chunk])
            if len(predictions) != len(chunk):
                raise RuntimeError("single-source batch output length mismatch")
            errors = [None] * len(chunk)
        except Exception:
            predictions, errors = [], []
            for row in chunk:
                try:
                    predictions.append(backend.transcribe(row["audio_path"]))
                    errors.append(None)
                except Exception as exc:
                    predictions.append({"hyp": "", "lang": "", "words": None})
                    errors.append(f"{type(exc).__name__}: {exc}")
        for row, prediction, error in zip(chunk, predictions, errors):
            append_jsonl(
                destination,
                _score_single_source(row, prediction, error, config["scoring"], model_name),
            )
        written += len(chunk)
        print(f"[{written}/{len(items)}] {time.time() - started:.1f}s", flush=True)
    return destination


def analyze_single_source(path: str | Path) -> dict[str, Any]:
    rows = list(read_jsonl(path))
    summary: dict[str, Any] = {"n_rows": len(rows), "sources": {}}
    for source_type in ("speech", "vocal"):
        values = [row for row in rows if row["source_type"] == source_type]
        valid = [row for row in values if row.get("error") is None]
        reference_words = sum(int(row["n_reference_words"]) for row in valid)
        edits = sum(
            int(row["substitutions"] + row["deletions"] + row["insertions"])
            for row in valid
        )
        summary["sources"][source_type] = {
            "n": len(values),
            "n_valid": len(valid),
            "error_rate": 1.0 - len(valid) / len(values) if values else None,
            "micro_wer": edits / reference_words if reference_words else None,
            "macro_wer": mean(float(row["wer"]) for row in valid if row.get("wer") is not None)
            if valid
            else None,
            "mean_reference_recall": mean(
                float(row["reference_recall"])
                for row in valid
                if row.get("reference_recall") is not None
            )
            if valid
            else None,
            "no_grounded_output_rate": mean(
                float(bool(row["no_grounded_output"])) for row in valid
            )
            if valid
            else None,
        }
    return summary


def write_single_source_summary(run_path: str | Path, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(analyze_single_source(run_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def _unique_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        speech_id = str(row["speech_id"])
        speech_sample = stable_id("single_source", "speech", speech_id)
        output[speech_sample] = {
            "sample_id": speech_sample,
            "source_type": "speech",
            "source_id": speech_id,
            "speaker_id": str(row["speech_speaker_id"]),
            "track_id": None,
            "audio_path": str(row["speech_path"]),
            "reference": str(row["speech_reference"]),
        }
        crop_id = str(row["vocal_crop_id"])
        vocal_sample = stable_id("single_source", "vocal", crop_id)
        output[vocal_sample] = {
            "sample_id": vocal_sample,
            "source_type": "vocal",
            "source_id": crop_id,
            "speaker_id": None,
            "track_id": str(row["track_id"]),
            "audio_path": str(row["vocal_path"]),
            "reference": str(row["lyric_reference"]),
        }
    return [output[key] for key in sorted(output)]


def _score_single_source(
    row: dict[str, Any],
    prediction: dict[str, Any],
    error: str | None,
    scoring: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    reference = normalize_words(row["reference"], scoring)
    hypothesis = normalize_words(str(prediction.get("hyp", "")), scoring)
    substitutions, deletions, insertions = word_error_counts(reference, hypothesis)
    if row["source_type"] == "speech":
        attribution = attribute_tokens(prediction.get("hyp", ""), row["reference"], "", scoring)
        recall = attribution["speech_exclusive_recall"]
    else:
        attribution = attribute_tokens(prediction.get("hyp", ""), "", row["reference"], scoring)
        recall = attribution["lyric_exclusive_recall"]
    return {
        **row,
        "model": model_name,
        **prediction,
        "error": error,
        "n_reference_words": len(reference),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": (substitutions + deletions + insertions) / len(reference)
        if reference
        else None,
        "reference_recall": recall,
        "n_grounded": attribution["n_grounded"],
        "no_grounded_output": attribution["no_grounded_output"],
    }


def word_error_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    # Each cell stores (total edits, substitutions, deletions, insertions).
    previous = [(index, 0, index, 0) for index in range(len(reference) + 1)]
    for hyp_index, hyp_word in enumerate(hypothesis, 1):
        current = [(hyp_index, 0, 0, hyp_index)]
        for ref_index, ref_word in enumerate(reference, 1):
            if ref_word == hyp_word:
                current.append(previous[ref_index - 1])
                continue
            substitution = previous[ref_index - 1]
            deletion = current[ref_index - 1]
            insertion = previous[ref_index]
            candidates = [
                (substitution[0] + 1, substitution[1] + 1, substitution[2], substitution[3]),
                (deletion[0] + 1, deletion[1], deletion[2] + 1, deletion[3]),
                (insertion[0] + 1, insertion[1], insertion[2], insertion[3] + 1),
            ]
            current.append(min(candidates))
        previous = current
    _, substitutions, deletions, insertions = previous[-1]
    return substitutions, deletions, insertions
