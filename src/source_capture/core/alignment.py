from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .models import NeMoCTCBackend
from .io import append_jsonl, read_jsonl


def align_run(
    run_path: str | Path,
    output_path: str | Path,
    model_config: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Force-align every complete hypothesis with one shared NeMo CTC model."""
    source = Path(run_path)
    destination = Path(output_path)
    if destination.exists() and overwrite:
        destination.unlink()
    completed = (
        {str(row["sample_id"]) for row in read_jsonl(destination)}
        if destination.exists()
        else set()
    )
    rows = [row for row in read_jsonl(source) if str(row["sample_id"]) not in completed]
    backend = NeMoCTCBackend(model_config)
    chunk_size = int(model_config.get("alignment_batch_size", 16))
    started = time.time()
    written = 0
    for offset in range(0, len(rows), chunk_size):
        chunk = rows[offset : offset + chunk_size]
        alignable = [row for row in chunk if not row.get("error") and str(row.get("hyp", "")).strip()]
        aligned: dict[str, dict[str, Any]] = {}
        if alignable:
            try:
                aligned = _align_batch(backend, alignable)
            except Exception:
                for row in alignable:
                    try:
                        aligned.update(_align_batch(backend, [row]))
                    except Exception as exc:
                        aligned[str(row["sample_id"])] = {
                            "words": None,
                            "alignment_error": f"{type(exc).__name__}: {exc}",
                        }
        for row in chunk:
            sample_id = str(row["sample_id"])
            if row.get("error"):
                result = {"words": None, "alignment_error": "source inference error"}
            elif not str(row.get("hyp", "")).strip():
                result = {"words": [], "alignment_error": None}
            else:
                result = aligned[sample_id]
            append_jsonl(
                destination,
                {
                    "sample_id": sample_id,
                    **result,
                    "aligner_model": str(model_config["model_name"]),
                    "aligner_revision": model_config.get("resolved_revision"),
                    "alignment_method": "NeMo CTC forced alignment (Viterbi)",
                    "output_timestep_s": backend.time_stride,
                },
            )
        written += len(chunk)
        print(f"[{written}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    return destination


def _align_batch(
    backend: NeMoCTCBackend, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    from nemo.collections.asr.parts.utils.aligner_utils import (
        add_t_start_end_to_utt_obj,
        get_batch_variables,
        viterbi_decoding,
    )

    audio = [str(row["audio_path"]) for row in rows]
    text = [str(row["hyp"]) for row in rows]
    variables = get_batch_variables(
        audio=audio,
        model=backend.model,
        segment_separators=[],
        word_separator=" ",
        align_using_pred_text=False,
        gt_text_batch=text,
        output_timestep_duration=backend.time_stride,
        verbose=False,
    )
    log_probs, targets, lengths, target_lengths, utterances, timestep = variables
    alignments = viterbi_decoding(
        log_probs,
        targets,
        lengths,
        target_lengths,
        viterbi_device=next(backend.model.parameters()).device,
    )
    output: dict[str, dict[str, Any]] = {}
    for row, utterance, alignment in zip(rows, utterances, alignments):
        timed = add_t_start_end_to_utt_obj(utterance, alignment, timestep)
        words = _extract_aligned_words(timed)
        expected = len(str(row["hyp"]).split())
        output[str(row["sample_id"])] = {
            "words": words,
            "alignment_error": (
                None
                if words
                else f"forced alignment returned no words for {expected} whitespace tokens"
            ),
        }
    return output


def _extract_aligned_words(utterance: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment_or_token in getattr(utterance, "segments_and_tokens", []):
        for word_or_token in getattr(segment_or_token, "words_and_tokens", []):
            if not hasattr(word_or_token, "tokens"):
                continue
            word = str(getattr(word_or_token, "text", "") or "").strip()
            start = getattr(word_or_token, "t_start", None)
            end = getattr(word_or_token, "t_end", None)
            if not word or start is None or end is None or float(start) < 0 or float(end) < 0:
                continue
            output.append({"word": word, "start": float(start), "end": float(end)})
    return output
