from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from ..config import experiment_dir
from ..core.io import read_jsonl
from ..core.scoring import score_time_window
from ..core.statistics import exact_sign_flip_pvalue, two_way_cluster_bootstrap
from ..core.summary import output_selection_summary


METRICS = (
    "lir",
    "speech_exclusive_recall",
    "lyric_exclusive_recall",
    "speech_shared_reference_rate",
    "lyric_shared_reference_rate",
    "no_grounded_output",
    # Secondary diagnostic metrics.
    "scs",
    "tsr",
)


def analyze_experiment(config: dict[str, Any], experiment: str) -> tuple[Path, Path]:
    directory = experiment_dir(config, experiment)
    score_files = sorted((directory / "scores").glob("*.jsonl"))
    if not score_files:
        raise FileNotFoundError(f"no score files below {directory / 'scores'}")
    rows = [row for path in score_files for row in read_jsonl(path)]
    summary = {
        "experiment": experiment,
        "models": sorted({row["model"] for row in rows}),
        "n_rows": len(rows),
        "aggregates": _aggregates(rows, experiment),
    }
    if experiment == "onset_position":
        summary["primary_contrasts"] = _onset_position_contrasts(rows, config)
        summary["adverse_snr_baseline_outputs"] = _adverse_snr_baseline_outputs(rows)
    elif experiment == "temporal_history":
        summary["history_effects"] = _temporal_history_effects(rows, config)
        summary["symmetric_order_effects"] = _temporal_history_symmetric_order_effects(
            rows, config
        )
    json_path = directory / "summary.json"
    csv_path = directory / "summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(csv_path, summary["aggregates"])
    return json_path, csv_path


def _adverse_snr_baseline_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize the -10 dB baseline used for the utterance-level claim."""
    output = []
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        selected = [
            row
            for row in rows
            if row["model"] == model
            and row["split"] == "evaluation"
            and row["condition"] == "baseline"
            and float(row["snr_db"]) == -10.0
        ]
        output.append({"model": model, **output_selection_summary(selected)})
    return output


def _aggregates(rows: list[dict[str, Any]], experiment: str) -> list[dict[str, Any]]:
    fields = {
        "onset_position": ["model", "split", "snr_db", "condition"],
        "temporal_history": ["model", "split", "condition", "switch_s"],
    }[experiment]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        aggregate = {field: value for field, value in zip(fields, key)}
        aggregate["n"] = len(values)
        aggregate["n_tracks"] = len({row["track_id"] for row in values})
        aggregate["n_utterances"] = len(
            {row.get("speech_id") for row in values if row.get("speech_id")}
        )
        scored = [row for row in values if row.get("score_error") is None]
        aggregate["n_scored"] = len(scored)
        aggregate["n_inference_errors"] = sum(bool(row.get("error")) for row in values)
        aggregate["n_missing_word_timestamps"] = sum(
            row.get("words") is None for row in scored
        )
        grounded = [int(row.get("n_grounded", 0)) for row in scored]
        aggregate["mean_n_grounded"] = mean(grounded) if grounded else None
        aggregate["median_n_grounded"] = (
            float(np.median(np.asarray(grounded))) if grounded else None
        )
        for metric in METRICS:
            available = [float(row[metric]) for row in scored if row.get(metric) is not None]
            aggregate[f"mean_{metric}"] = mean(available) if available else None
        output.append(aggregate)
    return output


def _onset_position_contrasts(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    comparators = {"middle", "offset", "distributed"}
    grouped: dict[tuple[str, str, float], dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        grouped[(row["model"], row["split"], float(row["snr_db"]))][row["pair_id"]][
            row["condition"]
        ] = row
    output = []
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    for (model, split, snr_db), pairs in sorted(grouped.items()):
        for metric in ("lir", "tsr"):
            track_effects: dict[str, list[float]] = defaultdict(list)
            for conditions in pairs.values():
                if "onset" not in conditions or not comparators <= set(conditions):
                    continue
                onset = conditions["onset"].get(metric)
                values = [conditions[name].get(metric) for name in comparators]
                if onset is None or any(value is None for value in values):
                    continue
                effect = float(onset) - mean(float(value) for value in values)
                track_effects[conditions["onset"]["track_id"]].append(effect)
            by_track = [mean(values) for values in track_effects.values()]
            estimate, low, high = _bootstrap_track_means(by_track, replicates, seed)
            output.append(
                {
                    "model": model,
                    "split": split,
                    "snr_db": snr_db,
                    "metric": metric,
                    "contrast": (
                        f"{metric.upper()}(onset)-"
                        f"mean({metric.upper()}(middle,offset,distributed))"
                    ),
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_tracks": len(by_track),
                }
            )
    return output


def _temporal_history_effects(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    by_key = {
        (row["model"], row["split"], row["pair_id"], row["condition_id"]): row
        for row in rows
    }
    pair_effects = []
    missing_alignment: list[str] = []
    for row in rows:
        comparator_id = row.get("static_comparator_condition")
        if row.get("condition") not in {"s_to_l", "l_to_s"} or not comparator_id:
            continue
        comparator = by_key.get((row["model"], row["split"], row["pair_id"], comparator_id))
        if comparator is None:
            continue
        row_alignment_missing = row.get("words") is None and bool(str(row.get("hyp", "")).strip())
        comparator_alignment_missing = comparator.get("words") is None and bool(
            str(comparator.get("hyp", "")).strip()
        )
        if row_alignment_missing or comparator_alignment_missing or not row.get(
            "speech_words"
        ) or not row.get("lyric_words"):
            missing_alignment.append(f"{row['model']}:{row['sample_id']}")
            continue
        start, end = float(row["evaluation_start_s"]), float(row["evaluation_end_s"])
        effect: dict[str, Any] = {
            "model": row["model"],
            "split": row["split"],
            "pair_id": row["pair_id"],
            "track_id": row["track_id"],
            "speech_id": row.get("speech_id"),
            "condition": row["condition"],
            "switch_s": row["switch_s"],
            "evaluation_start_s": start,
            "evaluation_end_s": end,
            "expected_direction": row["history_effect_direction"],
        }
        for label, margin in (("primary", 0.0), ("boundary_0p2", 0.2)):
            switched_window = score_time_window(
                row.get("words") or [],
                start,
                end,
                row["speech_reference"],
                row["lyric_reference"],
                config["scoring"],
                speech_words=row.get("speech_words"),
                lyric_words=row.get("lyric_words"),
                boundary_exclusion_s=margin,
            )
            comparator_window = score_time_window(
                comparator.get("words") or [],
                start,
                end,
                comparator["speech_reference"],
                comparator["lyric_reference"],
                config["scoring"],
                speech_words=comparator.get("speech_words"),
                lyric_words=comparator.get("lyric_words"),
                boundary_exclusion_s=margin,
            )
            assert switched_window is not None and comparator_window is not None
            for metric in (
                "lir",
                "scs",
                "speech_exclusive_recall",
                "lyric_exclusive_recall",
                "no_grounded_output",
            ):
                left = switched_window.get(metric)
                right = comparator_window.get(metric)
                prefix = "" if label == "primary" else f"{label}_"
                effect[f"{prefix}switched_{metric}"] = left
                effect[f"{prefix}comparator_{metric}"] = right
                effect[f"{prefix}{metric}_effect"] = (
                    float(left) - float(right)
                    if left is not None and right is not None
                    else None
                )
        # Clip-level signed source-selection effect.
        effect["history_effect"] = effect.get("scs_effect")
        pair_effects.append(effect)
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_effects:
        grouped[(row["model"], row["split"], row["condition"], float(row["switch_s"]))].append(row)
    aggregate = []
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    for (model, split, condition, switch_s), values in sorted(grouped.items()):
        for prefix in ("", "boundary_0p2_"):
            for metric in (
                "lir",
                "scs",
                "speech_exclusive_recall",
                "lyric_exclusive_recall",
                "no_grounded_output",
            ):
                key = f"{prefix}{metric}_effect"
                available = [value for value in values if value.get(key) is not None]
                summary = two_way_cluster_bootstrap(
                    available,
                    value_key=key,
                    replicates=replicates,
                    seed=seed,
                )
                by_track: dict[str, list[float]] = defaultdict(list)
                by_utterance: dict[str, list[float]] = defaultdict(list)
                for value in available:
                    by_track[str(value["track_id"])].append(float(value[key]))
                    by_utterance[str(value["speech_id"])].append(float(value[key]))
                summary.update(
                    {
                        "model": model,
                        "split": split,
                        "condition": condition,
                        "switch_s": switch_s,
                        "metric": metric,
                        "window_sensitivity": "primary" if not prefix else "drop_0.2s_boundaries",
                        "p_exact_track": exact_sign_flip_pvalue(
                            [mean(items) for items in by_track.values()]
                        ),
                        "p_exact_utterance": exact_sign_flip_pvalue(
                            [mean(items) for items in by_utterance.values()]
                        ),
                        "expected_direction": values[0]["expected_direction"],
                    }
                )
                aggregate.append(summary)
    if not pair_effects:
        raise RuntimeError(
            "no window-level history effect could be computed: every scored row is "
            "missing model word timestamps, aligned speech/lyric references, or its "
            "matched comparator. Check models.<name>.word_alignment_manifest together "
            "with prefer_external_word_alignment and require_external_word_alignment."
        )
    return {
        "aggregate": aggregate,
        "pair_effects": pair_effects,
        "missing_alignment_rows": len(missing_alignment),
        "missing_alignment_rate": (
            len(missing_alignment) / len(rows) if rows else None
        ),
        "missing_alignment_examples": missing_alignment[:10],
        "note": (
            "Rows without model output timestamps or aligned speech/lyric references "
            "are excluded from window-level history effects; clip-level scores remain valid."
        ),
    }


def _temporal_history_symmetric_order_effects(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("condition_id") in {"s5_to_l5", "l5_to_s5"}:
            grouped[(row["model"], row["split"], row["pair_id"])][
                row["condition_id"]
            ] = row
    by_group: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (model, split, _), conditions in grouped.items():
        if not {"s5_to_l5", "l5_to_s5"} <= set(conditions):
            continue
        left = conditions["s5_to_l5"].get("scs")
        right = conditions["l5_to_s5"].get("scs")
        if left is None or right is None:
            continue
        track = conditions["s5_to_l5"]["track_id"]
        by_group[(model, split)][track].append(float(left) - float(right))
    seed = int(config["project"]["seed"])
    replicates = int(config["scoring"].get("bootstrap_replicates", 10000))
    output = []
    for (model, split), tracks in sorted(by_group.items()):
        values = [mean(effects) for effects in tracks.values()]
        estimate, low, high = _bootstrap_track_means(values, replicates, seed)
        output.append(
            {
                "model": model,
                "split": split,
                "contrast": "SCS(S5->L5)-SCS(L5->S5)",
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_tracks": len(values),
            }
        )
    return output


def _bootstrap_track_means(
    values: list[float], replicates: int, seed: int
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(replicates, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
