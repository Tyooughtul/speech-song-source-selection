"""Onset suppression and speech-enhancement front-end experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml
from scipy.signal import resample_poly

from ..core.audio import peak, read_audio, read_pcm16, write_pcm16
from ..core.models import create_backend
from ..core.io import append_jsonl, read_jsonl, sha256_file, stable_id, write_jsonl
from ..core.scoring import attribute_tokens, empty_score


CONDITIONS = ("baseline", "oracle_onset", "actual_onset", "actual_full", "actual_end")


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Front-end config must be a mapping")
    return value


def prepare(config: dict[str, Any]) -> Path:
    source = Path(config["project"]["onset_position_root"]) / "manifest.jsonl"
    settings = config["selection"]
    snr = float(settings["snr_db"])
    rows = [
        row for row in read_jsonl(source)
        if row.get("split") == config["project"].get("split", "evaluation")
        and float(row["snr_db"]) == snr
        and row["condition"] in {"baseline", "onset"}
    ]
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = row
    by_track: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for conditions in by_pair.values():
        if {"baseline", "onset"} <= set(conditions):
            by_track[conditions["baseline"]["track_id"]].append(conditions)
    selected = []
    for track, pairs in sorted(by_track.items()):
        ordered = sorted(pairs, key=lambda value: stable_id("onset_mitigation", track, value["baseline"]["pair_id"]))
        selected.extend(ordered[: int(settings["pairs_per_track"])])
    manifest = []
    for pair in selected:
        baseline, oracle = pair["baseline"], pair["onset"]
        common = {
            key: baseline.get(key)
            for key in (
                "pair_id", "track_id", "speech_id", "speech_speaker_id", "split",
                "speech_reference", "lyric_reference", "lyric_reference_scope",
            )
        }
        for condition, source_row in (("baseline", baseline), ("oracle_onset", oracle)):
            manifest.append(
                {
                    **common,
                    "experiment": "front_end",
                    "condition": condition,
                    "sample_id": stable_id("onset_mitigation", baseline["pair_id"], condition),
                    "audio_path": source_row["audio_path"],
                    "audio_sha256": source_row["audio_sha256"],
                    "frontend": "none" if condition == "baseline" else "oracle_vocal_stem_ducking",
                    "processed_duration_s": 0.0 if condition == "baseline" else float(config["frontend"]["intervention_duration_s"]),
                    "processing_seconds": 0.0,
                    "rtf": 0.0,
                }
            )
    destination = Path(config["project"]["output_root"]) / "manifest.base.jsonl"
    write_jsonl(destination, manifest)
    return destination


def enhance(config: dict[str, Any], *, limit_pairs: int | None = None) -> Path:
    root = Path(config["project"]["output_root"])
    base_manifest = root / "manifest.base.jsonl"
    if not base_manifest.exists():
        prepare(config)
    base_rows = list(read_jsonl(base_manifest))
    baseline_rows = [row for row in base_rows if row["condition"] == "baseline"]
    if limit_pairs is not None:
        baseline_rows = baseline_rows[:limit_pairs]
    generated_path = root / "manifest.actual.jsonl"
    completed = {(row["pair_id"], row["condition"]) for row in read_jsonl(generated_path)} if generated_path.exists() else set()
    frontend = DeepFilterBinary(config["frontend"])
    for index, row in enumerate(baseline_rows, 1):
        audio, sample_rate = read_pcm16(row["audio_path"])
        duration = float(config["frontend"]["intervention_duration_s"])
        width = round(duration * sample_rate)
        segments = {
            "actual_onset": (0, width),
            "actual_full": (0, len(audio)),
            "actual_end": (len(audio) - width, len(audio)),
        }
        for condition, (start, end) in segments.items():
            if (row["pair_id"], condition) in completed:
                continue
            enhanced_segment, processing_seconds = frontend.enhance(audio[start:end], sample_rate)
            output = _splice(audio, enhanced_segment, start, end, sample_rate, float(config["frontend"]["splice_crossfade_ms"]))
            if peak(output) > 1.0:
                output = output / peak(output) * 0.999
            sample_id = stable_id("onset_mitigation", row["pair_id"], condition)
            path = root / "audio" / f"{sample_id}.wav"
            write_pcm16(path, output, sample_rate)
            processed_duration = (end - start) / sample_rate
            append_jsonl(
                generated_path,
                {
                    **row,
                    "condition": condition,
                    "sample_id": sample_id,
                    "audio_path": str(path.resolve()),
                    "audio_sha256": sha256_file(path),
                    "frontend": f"{config['frontend']['name']}-{config['frontend']['version']}",
                    "processed_duration_s": processed_duration,
                    "processing_seconds": processing_seconds,
                    "rtf": processing_seconds / processed_duration,
                },
            )
        print(f"[{index}/{len(baseline_rows)}]", flush=True)
    actual_rows = list(read_jsonl(generated_path)) if generated_path.exists() else []
    selected_pairs = {row["pair_id"] for row in baseline_rows}
    final_rows = [row for row in base_rows if row["pair_id"] in selected_pairs] + [row for row in actual_rows if row["pair_id"] in selected_pairs]
    destination = root / "manifest.jsonl"
    write_jsonl(destination, sorted(final_rows, key=lambda row: (row["track_id"], row["pair_id"], CONDITIONS.index(row["condition"]))))
    return destination


class DeepFilterBinary:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.binary = Path(config["binary_path"])
        if not self.binary.exists():
            raise FileNotFoundError(self.binary)
        self.sample_rate = int(config["sample_rate"])

    def enhance(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        input_48k = resample_poly(audio, self.sample_rate, sample_rate).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="front_end_df_") as directory:
            work = Path(directory)
            input_path = work / "input.wav"
            output_dir = work / "out"
            write_pcm16(input_path, input_48k, self.sample_rate)
            command = [str(self.binary), "--output-dir", str(output_dir)]
            if bool(self.config.get("compensate_delay", True)):
                command.append("--compensate-delay")
            command.append(str(input_path))
            started = time.perf_counter()
            process = subprocess.run(command, check=False, capture_output=True, text=True)
            elapsed = time.perf_counter() - started
            if process.returncode != 0:
                raise RuntimeError(f"DeepFilterNet failed: {process.stderr[-1000:]}")
            candidates = sorted(output_dir.glob("*.wav"))
            if len(candidates) != 1:
                raise RuntimeError(f"DeepFilterNet produced {len(candidates)} wav files")
            enhanced = read_audio(candidates[0], self.sample_rate)
        restored = resample_poly(enhanced, sample_rate, self.sample_rate).astype(np.float32)
        target = len(audio)
        if len(restored) < target:
            restored = np.pad(restored, (0, target - len(restored)))
        return restored[:target], elapsed


def _splice(base: np.ndarray, enhanced: np.ndarray, start: int, end: int, sample_rate: int, crossfade_ms: float) -> np.ndarray:
    output = base.copy()
    output[start:end] = enhanced[: end - start]
    fade = max(1, round(crossfade_ms * sample_rate / 1000.0))
    if start > 0:
        width = min(fade, start, end - start)
        weight = np.linspace(0.0, 1.0, width, dtype=np.float32)
        output[start : start + width] = base[start : start + width] * (1 - weight) + enhanced[:width] * weight
    if end < len(base):
        width = min(fade, len(base) - end, end - start)
        weight = np.linspace(1.0, 0.0, width, dtype=np.float32)
        output[end - width : end] = base[end - width : end] * (1 - weight) + enhanced[end - start - width : end - start] * weight
    return output


def _backend_config(config: dict[str, Any], model_name: str) -> dict[str, Any]:
    if model_name in config.get("models", {}):
        return dict(config["models"][model_name])
    if model_name == "whisper" and "model" in config:
        return {"backend": "whisper_hf", **config["model"]}
    raise ValueError(f"model is not configured for Front-end: {model_name}")


def _run_row(
    row: dict[str, Any],
    model_name: str,
    backend: Any,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    started = time.time()
    try:
        result = backend.transcribe(row["audio_path"])
        result["error"] = None
        score = attribute_tokens(
            result["hyp"],
            row["speech_reference"],
            row["lyric_reference"],
            scoring,
        )
        score["score_error"] = None
    except Exception as exc:
        result = {
            "hyp": "",
            "lang": "",
            "words": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
        score = empty_score("inference error")
    elapsed = time.time() - started
    duration = float(row.get("duration_s") or _audio_duration_s(row))
    return {
        **row,
        **result,
        **score,
        "model": model_name,
        "processing_seconds": elapsed,
        "rtf": elapsed / duration if duration > 0 else None,
    }


def _audio_duration_s(row: dict[str, Any]) -> float:
    audio, sample_rate = read_pcm16(row["audio_path"])
    return len(audio) / sample_rate


def run(
    config: dict[str, Any],
    *,
    model_name: str = "whisper",
    limit: int | None = None,
) -> Path:
    root = Path(config["project"]["output_root"])
    manifest = root / "manifest.jsonl"
    if not manifest.exists():
        enhance(config)
    destination = root / "runs" / f"{model_name}.jsonl"
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
            results = executor.map(task, rows)
            for index, result in enumerate(results, 1):
                append_jsonl(destination, result)
                if index % 10 == 0 or index == len(rows):
                    print(f"[{index}/{len(rows)}] {time.time() - started:.1f}s", flush=True)
    else:
        backend = create_backend(model_name, backend_config)
        for index, row in enumerate(rows, 1):
            append_jsonl(
                destination,
                _run_row(row, model_name, backend, config["scoring"]),
            )
            if index % 10 == 0 or index == len(rows):
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
    for condition in CONDITIONS:
        values = grouped.get(condition, [])
        record = {"condition": condition, "n": len(values), "n_tracks": len({row["track_id"] for row in values})}
        for metric in (
            "lir",
            "speech_exclusive_recall",
            "lyric_exclusive_recall",
            "no_grounded_output",
            "tsr",
            "scs",
            "rtf",
            "processing_seconds",
        ):
            available = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[f"mean_{metric}"] = mean(available) if available else None
        aggregates.append(record)
    contrasts = []
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_pair[row["pair_id"]][row["condition"]] = row
    for condition in CONDITIONS[1:]:
        for metric in (
            "lir",
            "speech_exclusive_recall",
            "lyric_exclusive_recall",
            "no_grounded_output",
            "tsr",
            "scs",
        ):
            effects = []
            for values in by_pair.values():
                if condition in values and "baseline" in values and values[condition].get(metric) is not None and values["baseline"].get(metric) is not None:
                    effects.append((values[condition]["track_id"], float(values[condition][metric]) - float(values["baseline"][metric])))
            contrasts.append(_contrast(condition, metric, effects, config))
    onset_share = _onset_share(by_pair)
    summary = {
        "experiment": "front_end",
        "model": model_name,
        "n_rows": len(all_rows),
        "n_scored": len(rows),
        "n_inference_errors": sum(bool(row.get("error")) for row in all_rows),
        "aggregates": aggregates,
        "contrasts": contrasts,
        "actual_onset_share_of_full_gain": onset_share,
        "frontend": config["frontend"],
        "interpretation_scope": "the tested DeepFilterNet2 front end only",
    }
    destination = root / ("summary.json" if model_name == "whisper" else f"summary.{model_name}.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _contrast(condition: str, metric: str, effects: list[tuple[str, float]], config: dict[str, Any]) -> dict[str, Any]:
    by_track: dict[str, list[float]] = defaultdict(list)
    for track, effect in effects:
        by_track[track].append(effect)
    values = np.asarray([mean(items) for items in by_track.values()])
    estimate = low = high = None
    if len(values):
        rng = np.random.default_rng(int(config["project"]["seed"]))
        draws = rng.choice(values, size=(int(config["scoring"]["bootstrap_replicates"]), len(values)), replace=True).mean(axis=1)
        estimate, low, high = float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return {"contrast": f"{metric}({condition})-{metric}(baseline)", "estimate": estimate, "ci95_low": low, "ci95_high": high, "n_pairs": len(effects), "n_tracks": len(values)}


def _onset_share(by_pair: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    output = {}
    for metric in ("lir", "tsr"):
        ratios = []
        for values in by_pair.values():
            if not {"baseline", "actual_onset", "actual_full"} <= set(values):
                continue
            baseline, onset, full = values["baseline"].get(metric), values["actual_onset"].get(metric), values["actual_full"].get(metric)
            if None in (baseline, onset, full) or abs(float(full) - float(baseline)) < 1e-6:
                continue
            ratios.append((float(onset) - float(baseline)) / (float(full) - float(baseline)))
        output[metric] = {"n": len(ratios), "mean_ratio": mean(ratios) if ratios else None, "median_ratio": float(np.median(ratios)) if ratios else None}
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Onset suppression and front-end experiment")
    parser.add_argument("command", choices=("prepare", "enhance", "run", "analyze", "all"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="whisper")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command in {"prepare", "all"}:
        print(prepare(config))
    if args.command in {"enhance", "all"}:
        print(enhance(config, limit_pairs=args.limit))
    if args.command in {"run", "all"}:
        print(run(config, model_name=args.model, limit=args.limit if args.command == "run" else None))
    if args.command in {"analyze", "all"}:
        print(analyze(config, model_name=args.model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
