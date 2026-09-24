from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .core.models import create_backend
from .config import experiment_dir
from .core.io import append_jsonl, load_json_or_jsonl, read_jsonl


def run_model(
    config: dict[str, Any],
    experiment: str,
    model_name: str,
    *,
    overwrite: bool = False,
    split: str = "all",
) -> Path:
    if model_name not in config["models"]:
        raise ValueError(f"model is not configured: {model_name}")
    manifest = experiment_dir(config, experiment) / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"generate {experiment} first: {manifest}")
    output = experiment_dir(config, experiment) / "runs" / f"{model_name}.jsonl"
    completed: set[str] = set()
    if output.exists() and not overwrite:
        completed = {row["sample_id"] for row in read_jsonl(output)}
    elif output.exists() and overwrite:
        output.unlink()
    rows = [
        row
        for row in read_jsonl(manifest)
        if model_name in row.get("eligible_models", [])
        and (split == "all" or row.get("split") == split)
        and row["sample_id"] not in completed
    ]
    backend_config = config["models"][model_name]
    external_alignments = _load_external_alignments(backend_config)
    workers = int(backend_config.get("workers", 1))
    if workers > 1 and backend_config.get("backend") == "qwen_http":
        _run_parallel(
            rows, output, model_name, backend_config, workers, external_alignments
        )
    elif backend_config.get("backend") == "nemo_ctc":
        backend = create_backend(model_name, backend_config)
        _run_batched(rows, output, model_name, backend, backend_config, external_alignments)
    else:
        backend = create_backend(model_name, backend_config)
        start = time.time()
        for index, row in enumerate(rows, 1):
            append_jsonl(
                output,
                _transcribe_one(backend, model_name, row, external_alignments),
            )
            if index % 50 == 0 or index == len(rows):
                elapsed = time.time() - start
                print(f"[{index}/{len(rows)}] {elapsed:.1f}s", flush=True)
    return output


def _run_batched(
    rows: list[dict[str, Any]],
    output: Path,
    model_name: str,
    backend: Any,
    backend_config: dict[str, Any],
    external_alignments: dict[str, list[dict[str, Any]]],
) -> None:
    chunk_size = int(backend_config.get("transcribe_chunk_size", 256))
    started = time.time()
    written = 0
    for offset in range(0, len(rows), chunk_size):
        chunk = rows[offset : offset + chunk_size]
        chunk_started = time.time()
        try:
            predictions = backend.transcribe_batch([row["audio_path"] for row in chunk])
            if len(predictions) != len(chunk):
                raise RuntimeError(
                    f"batch backend returned {len(predictions)} predictions for {len(chunk)} rows"
                )
            elapsed_each = (time.time() - chunk_started) / max(1, len(chunk))
            results = [
                _prediction_row(
                    row,
                    model_name,
                    prediction,
                    external_alignments,
                    elapsed_each,
                )
                for row, prediction in zip(chunk, predictions)
            ]
        except Exception:
            # Preserve row-level failure accounting and resumability if a large
            # chunk hits OOM or contains one malformed audio file.
            results = [
                _transcribe_one(backend, model_name, row, external_alignments)
                for row in chunk
            ]
        for result in results:
            append_jsonl(output, result)
        written += len(results)
        print(f"[{written}/{len(rows)}] {time.time() - started:.1f}s", flush=True)


def _run_parallel(
    rows: list[dict[str, Any]],
    output: Path,
    model_name: str,
    backend_config: dict[str, Any],
    workers: int,
    external_alignments: dict[str, list[dict[str, Any]]],
) -> None:
    # Each HTTP worker owns a tiny stateless backend instance. Results are
    # written in manifest order by executor.map.
    def task(row: dict[str, Any]) -> dict[str, Any]:
        backend = create_backend(model_name, backend_config)
        return _transcribe_one(backend, model_name, row, external_alignments)

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(task, rows), 1):
            append_jsonl(output, result)
            if index % 100 == 0 or index == len(rows):
                elapsed = time.time() - start
                print(f"[{index}/{len(rows)}] {elapsed:.1f}s", flush=True)


def _transcribe_one(
    backend: Any,
    model_name: str,
    row: dict[str, Any],
    external_alignments: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    start = time.time()
    try:
        prediction = backend.transcribe(row["audio_path"])
    except Exception as exc:
        prediction = {"hyp": "", "lang": "", "words": None}
        return {
            **row,
            "model": model_name,
            **prediction,
            "error": f"{type(exc).__name__}: {exc}",
            "inference_seconds": time.time() - start,
        }
    return _prediction_row(
        row,
        model_name,
        prediction,
        external_alignments,
        time.time() - start,
    )


def _prediction_row(
    row: dict[str, Any],
    model_name: str,
    prediction: dict[str, Any],
    external_alignments: dict[str, list[dict[str, Any]]],
    inference_seconds: float,
) -> dict[str, Any]:
    if not prediction.get("words") and row["sample_id"] in external_alignments:
        prediction["words"] = external_alignments[row["sample_id"]]
    return {
        **row,
        "model": model_name,
        **prediction,
        "error": None,
        "inference_seconds": inference_seconds,
    }


def _load_external_alignments(
    backend_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    path = backend_config.get("word_alignment_manifest")
    if not path or str(path).startswith("__SET_ME_"):
        return {}
    rows = load_json_or_jsonl(path)
    output: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if "sample_id" not in row or "words" not in row:
            raise ValueError("output alignment records need sample_id and words")
        if row.get("alignment_error") or row.get("words") is None:
            continue
        output[str(row["sample_id"])] = list(row["words"])
    return output
