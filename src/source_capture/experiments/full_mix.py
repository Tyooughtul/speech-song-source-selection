"""Full-mix accompaniment experiment for the primary evaluation set.

Motivation: Position/History/Scrambling mix the target speech with the isolated MUSDB18-HQ vocal
stem. Full-mix asks whether lyric capture persists when the track's original
accompaniment (drums + bass + other) is also present, i.e. when speech
competes with the full song rather than the isolated vocal.

Design: for every selected pair, two conditions share one per-pair
scale group:

- ``baseline_matched``: speech + g * vocal            (level-matched control)
- ``fullmix``:          speech + g * vocal + g * accomp

with g = vocal_gain_for_snr(speech, vocal, snr_db), exactly the Position baseline
gain at the same SNR. The accompaniment crop is taken from the track's
``accomp.wav`` at the pair's vocal-crop offset, so the vocal keeps its
natural balance against the accompaniment. Because both conditions share one
common scale, any fullmix-vs-baseline_matched difference is attributable to
the accompaniment alone: same speech, same vocal level, same overall level.
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

from ..core.audio import common_scale, peak, read_audio, read_pcm16, rms, vocal_gain_for_snr, write_pcm16
from ..core.models import create_backend
from ..core.io import append_jsonl, read_jsonl, sha256_file, stable_id, write_jsonl
from ..core.scoring import attribute_tokens, empty_score
from ..core.statistics import sign_flip_pvalue, two_way_cluster_bootstrap
from ..core.summary import output_type, source_transitions


CONDITIONS = ("baseline_matched", "fullmix")
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
        raise ValueError("Full-mix config must be a mapping")
    return config


def _reference_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in (
            "pair_id", "track_id", "vocal_crop_id", "speech_id", "speech_speaker_id",
            "speech_reference", "lyric_reference", "lyric_reference_scope", "split",
        )
    }


def _load_accompaniment(row: dict[str, Any], sample_rate: int, width: int) -> np.ndarray:
    directory = Path(row["vocal_source_path"]).parent
    candidates = (directory / "accomp.wav", directory / "accompaniment.wav")
    source = next((path for path in candidates if path.exists()), candidates[0])
    audio = read_audio(source, sample_rate)
    start = round(float(row["vocal_crop_start_s"]) * sample_rate)
    crop = audio[start : start + width]
    if len(crop) != width:
        raise ValueError(f"short accompaniment crop: {source}")
    return crop


def _mixtures(
    speech: np.ndarray,
    vocal: np.ndarray,
    accompaniment: np.ndarray,
    snr_db: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Unscaled condition mixtures and the Position-baseline vocal gain g.

    The arithmetic mirrors audio.compose_unscaled exactly (float32 gain
    envelope, float64 accumulation, float32 result), so that whenever the
    per-pair common scale is 1.0 the baseline_matched mixture is bit-identical
    to the Position baseline stimulus for the same pair.
    """
    gain = vocal_gain_for_snr(speech, vocal, snr_db)
    envelope = np.full(len(speech), gain, dtype=np.float32)
    speech64 = speech.astype(np.float64)
    vocal64 = vocal.astype(np.float64) * envelope
    return {
        "baseline_matched": (speech64 + vocal64).astype(np.float32),
        "fullmix": (speech64 + vocal64 + accompaniment.astype(np.float64) * envelope).astype(np.float32),
    }, gain


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
    snr_db = float(config["audio"]["snr_db"])
    headroom = float(config["audio"]["headroom_peak"])

    by_crop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_crop[row["vocal_crop_id"]].append(row)

    manifest: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for crop_id, pair_rows in sorted(by_crop.items()):
        exemplar = pair_rows[0]
        vocal, vocal_sr = read_pcm16(exemplar["vocal_path"])
        if vocal_sr != sample_rate or len(vocal) != expected_samples:
            raise ValueError(f"unexpected prepared vocal shape for {crop_id}")
        accompaniment = _load_accompaniment(exemplar, sample_rate, expected_samples)
        accompaniment_ratio = rms(accompaniment) / rms(vocal)
        for pair in pair_rows:
            speech, speech_sr = read_pcm16(pair["speech_path"])
            if speech_sr != sample_rate or len(speech) != expected_samples:
                raise ValueError(f"unexpected prepared speech shape for {pair['pair_id']}")
            mixtures, gain = _mixtures(speech, vocal, accompaniment, snr_db)
            scale = common_scale(list(mixtures.values()), headroom)
            for condition, mixture in mixtures.items():
                sample_id = stable_id("full_accompaniment", pair["pair_id"], condition)
                path = audio_root / f"{sample_id}.wav"
                write_pcm16(path, mixture * scale, sample_rate)
                manifest.append(
                    {
                        **_reference_fields(pair),
                        "experiment": "full_mix",
                        "condition": condition,
                        "sample_id": sample_id,
                        "family_id": stable_id("full_accompaniment", pair["pair_id"]),
                        "audio_path": str(path.resolve()),
                        "audio_sha256": sha256_file(path),
                        "snr_db": snr_db,
                        "vocal_gain": gain,
                        "accompaniment_rms_over_vocal_rms": accompaniment_ratio,
                        "common_scale": scale,
                    }
                )
            checks.append(
                {
                    "pair_id": pair["pair_id"],
                    "vocal_crop_id": crop_id,
                    "vocal_gain": gain,
                    "accompaniment_rms_over_vocal_rms": accompaniment_ratio,
                    "common_scale": scale,
                    "unscaled_peak": {name: peak(mix) for name, mix in mixtures.items()},
                    "final_peak": {name: peak(mix * scale) for name, mix in mixtures.items()},
                    "final_speech_rms": rms(speech * scale),
                    "final_vocal_rms": rms(vocal * gain * scale),
                    "final_accompaniment_rms": rms(accompaniment * gain * scale),
                    "clipped": bool(max(peak(mix * scale) for mix in mixtures.values()) > 1.0 + 1e-7),
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
    raise ValueError(f"model is not configured for Full-mix: {model_name}")


def _score(row: dict[str, Any], result: dict[str, Any], scoring: dict[str, Any]) -> dict[str, Any]:
    if result.get("error"):
        return empty_score("inference error")
    score = attribute_tokens(result["hyp"], row["speech_reference"], row["lyric_reference"], scoring)
    score["score_error"] = None
    return score


def _run_row(
    row: dict[str, Any],
    model_name: str,
    backend: Any,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    try:
        result = backend.transcribe(row["audio_path"])
        result["error"] = None
    except Exception as exc:
        result = {
            "hyp": "",
            "lang": "",
            "words": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {**row, **result, **_score(row, result, scoring), "model": model_name}


def run(
    config: dict[str, Any],
    *,
    model_name: str = "whisper",
    limit: int | None = None,
) -> Path:
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
        task = lambda row: _run_row(
            row,
            model_name,
            create_backend(model_name, backend_config),
            config["scoring"],
        )
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


def analyze(config: dict[str, Any], *, model_name: str = "whisper") -> Path:
    root = Path(config["project"]["output_root"])
    all_rows = list(read_jsonl(root / "runs" / f"{model_name}.jsonl"))
    rows = [row for row in all_rows if row.get("score_error") is None]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    aggregates = []
    for condition, values in sorted(grouped.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "n": len(values),
            "n_tracks": len({row["track_id"] for row in values}),
            "n_utterances": len({row["speech_id"] for row in values}),
        }
        for metric in CONTRAST_METRICS + ("n_grounded",):
            available = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[f"mean_{metric}"] = mean(available) if available else None
        record["mean_no_grounded_output"] = mean(
            float(bool(row.get("no_grounded_output"))) for row in values
        ) if values else None
        types = defaultdict(int)
        transitions: list[int] = []
        for row in values:
            category = output_type(row)
            types[category] += 1
            if category == "mixed":
                value = source_transitions(row)
                if value is not None:
                    transitions.append(value)
        total = max(1, len(values))
        record["output_type_rate"] = {
            name: types[name] / total
            for name in ("pure_speech", "pure_lyric", "mixed", "no_grounded")
        }
        record["mean_source_transitions_mixed"] = mean(transitions) if transitions else None
        aggregates.append(record)

    contrasts = _paired_contrasts(rows, config)
    scales = [float(row["common_scale"]) for row in rows]
    ratios = [float(row["accompaniment_rms_over_vocal_rms"]) for row in rows]
    summary = {
        "experiment": "full_mix",
        "model": model_name,
        "n_rows": len(all_rows),
        "n_scored": len(rows),
        "n_inference_errors": sum(bool(row.get("error")) for row in all_rows),
        "aggregates": aggregates,
        "contrasts": contrasts,
        "design": {
            "snr_db": float(config["audio"]["snr_db"]),
            "contrast": "fullmix - baseline_matched (same pair, same vocal gain, same common scale)",
            "common_scale": {"mean": mean(scales), "min": min(scales), "max": max(scales)},
            "accompaniment_rms_over_vocal_rms": {"mean": mean(ratios), "min": min(ratios), "max": max(ratios)},
        },
    }
    destination = root / ("summary.json" if model_name == "whisper" else f"summary.{model_name}.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _paired_contrasts(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = row
    output = []
    for metric in CONTRAST_METRICS:
        effects: list[dict[str, Any]] = []
        for conditions in by_pair.values():
            if "baseline_matched" not in conditions or "fullmix" not in conditions:
                continue
            a = conditions["fullmix"].get(metric)
            b = conditions["baseline_matched"].get(metric)
            if a is not None and b is not None:
                effects.append(
                    {
                        "track_id": conditions["fullmix"]["track_id"],
                        "speech_id": conditions["fullmix"].get("speech_id"),
                        "effect": float(a) - float(b),
                    }
                )
        output.append(_bootstrap_record(metric, effects, config))
    return output


def _bootstrap_record(metric: str, effects: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_track: dict[str, list[float]] = defaultdict(list)
    by_utterance: dict[str, list[float]] = defaultdict(list)
    for row in effects:
        by_track[str(row["track_id"])].append(float(row["effect"]))
        by_utterance[str(row["speech_id"])].append(float(row["effect"]))
    summary = two_way_cluster_bootstrap(
        effects,
        value_key="effect",
        replicates=int(config["scoring"]["bootstrap_replicates"]),
        seed=int(config["project"]["seed"]),
    )
    return {
        "contrast": f"{metric}(fullmix)-{metric}(baseline_matched)",
        **summary,
        "n_pairs": len(effects),
        "p_exact_track": sign_flip_pvalue([mean(items) for items in by_track.values()]),
        "p_exact_utterance": sign_flip_pvalue([mean(items) for items in by_utterance.values()]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full-accompaniment experiment")
    parser.add_argument("command", choices=("prepare", "run", "analyze"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="whisper")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "prepare":
        path = prepare(config)
    elif args.command == "run":
        path = run(config, model_name=args.model, limit=args.limit)
    else:
        path = analyze(config, model_name=args.model)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
