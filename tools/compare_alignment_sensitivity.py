#!/usr/bin/env python3
"""Compare windowed history effects under native and common alignment.

The history contrast is recomputed once with each model's own timestamps and
once with the single common NeMo CTC alignment. The result is alignment-stable
when the speech- and lyric-exclusive recall effects keep their direction.

Example:
    python tools/compare_alignment_sensitivity.py \
        --config configs/primary.local.yaml \
        --run outputs/primary/temporal_history/runs/whisper.jsonl \
        --common-alignment outputs/primary/common_alignment/whisper.jsonl \
        --output outputs/primary/alignment_sensitivity.whisper.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from source_capture.config import load_config
from source_capture.core.io import read_jsonl
from source_capture.experiments.position_history_analysis import _temporal_history_effects


def _key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["model"],
        row["split"],
        row["condition"],
        float(row["switch_s"]),
        row["metric"],
        row["window_sensitivity"],
    )


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "aggregate": result["aggregate"],
        "n_pair_effects": len(result["pair_effects"]),
        "missing_alignment_rows": result["missing_alignment_rows"],
        "missing_alignment_examples": result["missing_alignment_examples"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--common-alignment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()

    config = load_config(args.config, require_paths=False)
    config["scoring"]["bootstrap_replicates"] = args.bootstrap_replicates
    rows = list(read_jsonl(args.run))
    alignments = {
        str(row["sample_id"]): row for row in read_jsonl(args.common_alignment)
    }
    common_rows = []
    for row in rows:
        alignment = alignments.get(str(row["sample_id"]))
        common_rows.append(
            {
                **row,
                "native_words": row.get("words"),
                "words": (
                    alignment.get("words")
                    if alignment and not alignment.get("alignment_error")
                    else None
                ),
            }
        )
    native = _temporal_history_effects(rows, config)
    common = _temporal_history_effects(common_rows, config)
    native_by_key = {_key(row): row for row in native["aggregate"]}
    common_by_key = {_key(row): row for row in common["aggregate"]}
    comparisons = []
    for key in sorted(set(native_by_key) | set(common_by_key), key=str):
        left = native_by_key.get(key, {})
        right = common_by_key.get(key, {})
        native_estimate = left.get("estimate")
        common_estimate = right.get("estimate")
        comparisons.append(
            {
                "model": key[0],
                "split": key[1],
                "condition": key[2],
                "switch_s": key[3],
                "metric": key[4],
                "window_sensitivity": key[5],
                "native_estimate": native_estimate,
                "common_estimate": common_estimate,
                "same_sign": (
                    native_estimate is not None
                    and common_estimate is not None
                    and (float(native_estimate) == 0.0) == (float(common_estimate) == 0.0)
                    and float(native_estimate) * float(common_estimate) >= 0.0
                ),
            }
        )
    primary = [
        row
        for row in comparisons
        if row["condition"] == "s_to_l"
        and row["switch_s"] == 2.0
        and row["window_sensitivity"] == "primary"
        and row["metric"]
        in {"speech_exclusive_recall", "lyric_exclusive_recall", "lir", "no_grounded_output"}
    ]
    output = {
        "alignment_rows": len(alignments),
        "alignment_errors": sum(
            bool(row.get("alignment_error")) for row in alignments.values()
        ),
        "native": _compact(native),
        "common": _compact(common),
        "primary_2s_s_to_l": primary,
        "all_comparisons": comparisons,
        "decision_rule": (
            "The history contrast is alignment-stable only if the speech- and "
            "lyric-exclusive recall effects retain their directions under native "
            "and common alignment."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
