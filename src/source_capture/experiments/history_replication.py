"""Primary temporal-history contrast on the independent replication set.

Replicates exactly the History primary condition pair on a second, independently
prepared set of pairs:

- ``s_to_l``: tau=2 s of speech-dominant history (+10 dB) followed by a
  lyric-dominant segment (-10 dB);
- ``matched_static_lyric``: constant vocal gain equal to the s_to_l
  post-switch gain, computed on the evaluated-window slice, so the evaluated
  window [2.025, 4.025] s is acoustically identical between conditions.

Envelopes use the same helpers and arithmetic as the History builder in
position_history.py
(piecewise_gain_envelope, gains from vocal_gain_for_snr on the same slices,
50 ms raised-cosine ramps, one per-pair common scale).

Pipeline: prepare -> run (three backends) -> align (common NeMo CTC forced
alignment) -> analyze (window scoring, paired contrast, two-axis exact
sign-flip tests, two-way cluster bootstrap, intersection-union p).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml

from ..core.audio import common_scale, peak, piecewise_gain_envelope, read_pcm16, rms, vocal_gain_for_snr, write_pcm16
from ..core.models import create_backend
from ..core.alignment import align_run
from ..core.io import append_jsonl, read_jsonl, sha256_file, stable_id, write_jsonl
from ..core.scoring import attribute_tokens, empty_score, score_time_window
from ..core.statistics import sign_flip_pvalue, two_way_cluster_bootstrap

CONDITIONS = ("matched_static_lyric", "s_to_l")
CONTRAST_METRICS = (
    "lir",
    "speech_exclusive_recall",
    "lyric_exclusive_recall",
    "no_grounded_output",
    "tsr",
    "scs",
)


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("History replication config must be a mapping")
    return config


def _reference_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "pair_id", "track_id", "vocal_crop_id", "speech_id", "speech_speaker_id",
            "speech_reference", "lyric_reference", "lyric_reference_scope", "split",
            "speech_words", "lyric_words",
        )
    }


def _envelopes(
    config: dict[str, Any], speech: np.ndarray, vocal: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    params = config["experiment"]
    sample_rate = int(config["audio"]["sample_rate"])
    clip_s = float(config["audio"]["clip_duration_s"])
    ramp_ms = float(config["audio"]["intervention_ramp_ms"])
    tau = float(params["history_s"])
    speech_snr = float(params["speech_dominant_snr_db"])
    lyric_snr = float(params["lyric_dominant_snr_db"])
    evaluation_s = float(params["post_switch_evaluation_s"])
    evaluation_end = min(clip_s, tau + ramp_ms / 2000.0 + evaluation_s)

    pre_speech_gain = vocal_gain_for_snr(
        speech[: round(tau * sample_rate)], vocal[: round(tau * sample_rate)], speech_snr
    )
    post_slice = slice(
        round((tau + ramp_ms / 2000.0) * sample_rate), round(evaluation_end * sample_rate)
    )
    post_lyric_gain = vocal_gain_for_snr(speech[post_slice], vocal[post_slice], lyric_snr)
    envelopes = {
        "s_to_l": piecewise_gain_envelope(
            len(speech),
            sample_rate,
            [(0.0, tau, pre_speech_gain), (tau, clip_s, post_lyric_gain)],
            ramp_ms,
        ),
        "matched_static_lyric": np.full(len(speech), post_lyric_gain, dtype=np.float32),
    }
    gains = {"pre_speech_gain": pre_speech_gain, "post_lyric_gain": post_lyric_gain}
    return envelopes, gains


def prepare(config: dict[str, Any]) -> Path:
    source = Path(config["project"]["pairs_manifest"])
    rows = [
        row for row in read_jsonl(source)
        if row.get("split") == config["project"].get("split", "evaluation")
    ]
    output_root = Path(config["project"]["output_root"])
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    sample_rate = int(config["audio"]["sample_rate"])
    duration = float(config["audio"]["clip_duration_s"])
    expected_samples = round(sample_rate * duration)
    headroom = float(config["audio"]["headroom_peak"])

    manifest: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for pair in rows:
        speech, speech_sr = read_pcm16(pair["speech_path"])
        vocal, vocal_sr = read_pcm16(pair["vocal_path"])
        if speech_sr != sample_rate or vocal_sr != sample_rate:
            raise ValueError(f"sample-rate mismatch for {pair['pair_id']}")
        if len(speech) != expected_samples or len(vocal) != expected_samples:
            raise ValueError(f"unexpected prepared audio shape for {pair['pair_id']}")
        envelopes, gains = _envelopes(config, speech, vocal)
        mixtures = {
            name: (speech.astype(np.float64) + vocal.astype(np.float64) * env.astype(np.float64)).astype(np.float32)
            for name, env in envelopes.items()
        }
        scale = common_scale(list(mixtures.values()), headroom)
        for condition, mixture in mixtures.items():
            sample_id = stable_id("temporal_history_replication", pair["pair_id"], condition)
            path = audio_root / f"{sample_id}.wav"
            write_pcm16(path, mixture * scale, sample_rate)
            manifest.append(
                {
                    **_reference_fields(pair),
                    "experiment": "temporal_history_replication",
                    "condition": condition,
                    "sample_id": sample_id,
                    "family_id": stable_id("temporal_history_replication", pair["pair_id"]),
                    "audio_path": str(path.resolve()),
                    "audio_sha256": sha256_file(path),
                    "switch_s": float(config["experiment"]["history_s"]),
                    "evaluation_start_s": float(config["experiment"]["history_s"]) + float(config["audio"]["intervention_ramp_ms"]) / 2000.0,
                    "evaluation_end_s": float(config["experiment"]["history_s"]) + float(config["audio"]["intervention_ramp_ms"]) / 2000.0 + float(config["experiment"]["post_switch_evaluation_s"]),
                    "common_scale": scale,
                    **gains,
                }
            )
        checks.append(
            {
                "pair_id": pair["pair_id"],
                "common_scale": scale,
                "unscaled_peak": {name: peak(mix) for name, mix in mixtures.items()},
                "final_peak": {name: peak(mix * scale) for name, mix in mixtures.items()},
                **gains,
            }
        )
    destination = output_root / "manifest.jsonl"
    write_jsonl(destination, manifest)
    (output_root / "audio_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def _backend_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_name in config.get("models", {}):
        return dict(config["models"][model_name])
    raise ValueError(f"model is not configured for History replication: {model_name}")


def _score(row: dict[str, Any], result: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    if result.get("error"):
        return empty_score("inference error")
    score = attribute_tokens(result["hyp"], row["speech_reference"], row["lyric_reference"], scoring)
    score["score_error"] = None
    return score


def _run_row(row: dict[str, Any], model_name: str, backend: Any, scoring: dict[str, Any]) -> dict[str, Any]:
    try:
        result = backend.transcribe(row["audio_path"])
        result["error"] = None
    except Exception as exc:
        result = {"hyp": "", "lang": "", "words": None, "error": f"{type(exc).__name__}: {exc}"}
    return {**row, **result, **_score(row, result, scoring), "model": model_name}


def run(config: dict[str, Any], *, model_name: str = "whisper", limit: int | None = None) -> Path:
    output_root = Path(config["project"]["output_root"])
    manifest = output_root / "manifest.jsonl"
    if not manifest.exists():
        prepare(config)
    destination = output_root / "runs" / f"{model_name}.jsonl"
    completed = {row["sample_id"] for row in read_jsonl(destination)} if destination.exists() else set()
    rows = [row for row in read_jsonl(manifest) if row["sample_id"] not in completed]
    if limit is not None:
        rows = rows[:limit]
    backend_config = _backend_config(config, model_name)
    workers = int(backend_config.get("workers", 1))
    started = time.time()
    if workers > 1:
        task = lambda row: _run_row(row, model_name, create_backend(model_name, backend_config), config["scoring"])
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(task, rows), 1):
                append_jsonl(destination, result)
                if index % 20 == 0 or index == len(rows):
                    print(f"[{index}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    elif backend_config.get("backend") == "nemo_ctc":
        backend = create_backend(model_name, backend_config)
        chunk_size = int(backend_config.get("transcribe_chunk_size", 256))
        written = 0
        for offset in range(0, len(rows), chunk_size):
            chunk = rows[offset : offset + chunk_size]
            try:
                predictions = backend.transcribe_batch([row["audio_path"] for row in chunk])
                results = [
                    {**row, **prediction, "error": None,
                     **_score(row, {**prediction, "error": None}, config["scoring"]),
                     "model": model_name}
                    for row, prediction in zip(chunk, predictions)
                ]
            except Exception:
                results = [_run_row(row, model_name, backend, config["scoring"]) for row in chunk]
            for result in results:
                append_jsonl(destination, result)
            written += len(results)
            print(f"[{written}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    else:
        backend = create_backend(model_name, backend_config)
        for index, row in enumerate(rows, 1):
            append_jsonl(destination, _run_row(row, model_name, backend, config["scoring"]))
            if index % 20 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    return destination


def align(config: dict[str, Any], *, model_name: str) -> Path:
    output_root = Path(config["project"]["output_root"])
    destination = output_root / "common_alignment" / f"{model_name}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_config = dict(config["models"]["ctc"])
    model_config["alignment_batch_size"] = int(config.get("alignment", {}).get("batch_size", 16))
    return align_run(output_root / "runs" / f"{model_name}.jsonl", destination, model_config)


def analyze(config: dict[str, Any], *, model_name: str) -> Path:
    root = Path(config["project"]["output_root"])
    run_rows = {row["sample_id"]: row for row in read_jsonl(root / "runs" / f"{model_name}.jsonl")}
    aligned = {row["sample_id"]: row for row in read_jsonl(root / "common_alignment" / f"{model_name}.jsonl")}
    start_s = float(next(iter(run_rows.values()))["evaluation_start_s"])
    end_s = float(next(iter(run_rows.values()))["evaluation_end_s"])

    scored: list[dict[str, Any]] = []
    n_excluded = 0
    for sample_id, row in run_rows.items():
        alignment = aligned.get(sample_id)
        if alignment is None or alignment.get("words") is None or row.get("score_error") is not None:
            n_excluded += 1
            continue
        window_score = score_time_window(
            alignment["words"],
            start_s,
            end_s,
            row["speech_reference"],
            row["lyric_reference"],
            config["scoring"],
            speech_words=row.get("speech_words"),
            lyric_words=row.get("lyric_words"),
        )
        if window_score is None:
            n_excluded += 1
            continue
        scored.append({**row, **{f"window_{key}": value for key, value in window_score.items()}})

    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        by_pair[row["pair_id"]][row["condition"]] = row

    aggregates = []
    for condition in CONDITIONS:
        values = [row for rows in by_pair.values() for row in [rows.get(condition)] if row]
        record: dict[str, Any] = {"condition": condition, "n": len(values)}
        for metric in CONTRAST_METRICS:
            available = [float(row[f"window_{metric}"]) for row in values if row.get(f"window_{metric}") is not None]
            record[f"mean_{metric}"] = mean(available) if available else None
        aggregates.append(record)

    contrasts = []
    for metric in CONTRAST_METRICS:
        effects = []
        for conditions in by_pair.values():
            if "s_to_l" not in conditions or "matched_static_lyric" not in conditions:
                continue
            a = conditions["s_to_l"].get(f"window_{metric}")
            b = conditions["matched_static_lyric"].get(f"window_{metric}")
            if a is not None and b is not None:
                effects.append(
                    {
                        "track_id": conditions["s_to_l"]["track_id"],
                        "speech_id": conditions["s_to_l"]["speech_id"],
                        "effect": float(a) - float(b),
                    }
                )
        by_track: dict[str, list[float]] = defaultdict(list)
        by_utterance: dict[str, list[float]] = defaultdict(list)
        for effect in effects:
            by_track[effect["track_id"]].append(effect["effect"])
            by_utterance[effect["speech_id"]].append(effect["effect"])
        summary = two_way_cluster_bootstrap(
            effects,
            value_key="effect",
            replicates=int(config["scoring"]["bootstrap_replicates"]),
            seed=int(config["project"]["seed"]),
        )
        contrasts.append(
            {
                "contrast": f"window_{metric}(s_to_l)-window_{metric}(matched_static_lyric)",
                **summary,
                "n_pairs": len(effects),
                "p_exact_track": sign_flip_pvalue([mean(v) for v in by_track.values()]),
                "p_exact_utterance": sign_flip_pvalue([mean(v) for v in by_utterance.values()]),
            }
        )
    recall = {item["contrast"].split("(")[0]: item for item in contrasts}
    iu = {
        "endpoint": "intersection-union over co-primary window recalls",
        "p_exact_track": max(
            recall["window_speech_exclusive_recall"]["p_exact_track"] or 1.0,
            recall["window_lyric_exclusive_recall"]["p_exact_track"] or 1.0,
        ),
        "p_exact_utterance": max(
            recall["window_speech_exclusive_recall"]["p_exact_utterance"] or 1.0,
            recall["window_lyric_exclusive_recall"]["p_exact_utterance"] or 1.0,
        ),
    }
    summary = {
        "experiment": "temporal_history_replication",
        "model": model_name,
        "window_s": [start_s, end_s],
        "n_rows": len(run_rows),
        "n_scored_window": len(scored),
        "n_excluded": n_excluded,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "intersection_union": iu,
    }
    destination = root / ("summary.json" if model_name == "whisper" else f"summary.{model_name}.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Temporal-history replication experiment")
    parser.add_argument("command", choices=("prepare", "run", "align", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="whisper")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        path = prepare(config)
    elif args.command == "run":
        path = run(config, model_name=args.model, limit=args.limit)
    elif args.command == "align":
        path = align(config, model_name=args.model)
    else:
        path = analyze(config, model_name=args.model)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
