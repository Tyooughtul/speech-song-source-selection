"""Assemble the pre-specified four-hypothesis confirmatory family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .core.statistics import holm_adjust


MODELS = ("whisper", "qwen3")
SCRAMBLING_CONTRAST = (
    "lyric_exclusive_recall(shuffle_500ms)-"
    "lyric_exclusive_recall(intact)"
)


def confirmatory_family(
    history_summary: dict[str, Any],
    scrambling_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the paper's intersection-union and Holm correction rules.

    A p-value is carried as ``None`` when the underlying test could not be run,
    for example when an evaluation set has more clusters than exact enumeration allows.
    ``None`` propagates through the correction instead of raising.
    """
    by_model = {str(summary["model"]): summary for summary in scrambling_summaries}
    absent = [model for model in MODELS if model not in by_model]
    if absent:
        raise ValueError(
            "missing scrambling summary for: "
            + ", ".join(absent)
            + f" (got: {', '.join(sorted(by_model)) or 'none'})"
        )

    hypotheses: list[dict[str, Any]] = []
    history_rows = history_summary["history_effects"]["aggregate"]
    for model in MODELS:
        rows = [
            row
            for row in history_rows
            if row["model"] == model
            and row["condition"] == "s_to_l"
            and float(row["switch_s"]) == 2.0
            and row["window_sensitivity"] == "primary"
            and row["metric"] in {"speech_exclusive_recall", "lyric_exclusive_recall"}
        ]
        if len(rows) != 2:
            raise ValueError(
                f"expected two co-primary history rows for {model}, found {len(rows)}. "
                "An empty aggregate usually means the scoring stage dropped every word "
                "timestamp; check the history summary before assembling this family."
            )
        hypotheses.append(
            {
                "experiment": "history",
                "model": model,
                "endpoint": "intersection_union",
                "p_track_raw": _worst(row["p_exact_track"] for row in rows),
                "p_utterance_raw": _worst(row["p_exact_utterance"] for row in rows),
            }
        )

    for model in MODELS:
        rows = [
            row
            for row in by_model[model]["contrasts"]
            if row["contrast"] == SCRAMBLING_CONTRAST
        ]
        if len(rows) != 1:
            available = sorted({row["contrast"] for row in by_model[model]["contrasts"]})
            raise ValueError(
                f"expected one 500 ms scrambling contrast for {model}, found {len(rows)}. "
                f"Available contrasts: {', '.join(available) or 'none'}"
            )
        row = rows[0]
        hypotheses.append(
            {
                "experiment": "scrambling",
                "model": model,
                "endpoint": "lyric_exclusive_recall",
                "p_track_raw": _optional_float(row["p_exact_track"]),
                "p_utterance_raw": _optional_float(row["p_exact_utterance"]),
            }
        )

    track = holm_adjust([row["p_track_raw"] for row in hypotheses])
    utterance = holm_adjust([row["p_utterance_raw"] for row in hypotheses])
    for row, p_track, p_utterance in zip(hypotheses, track, utterance):
        row["p_track_holm"] = p_track
        row["p_utterance_holm"] = p_utterance
    return {
        "family": "history and scrambling; Whisper and Qwen3-ASR",
        "correction": "Holm correction applied separately on track and utterance axes",
        "null_p_semantics": "null marks an endpoint whose test was not run, e.g. a cluster count above the exact-enumeration budget",
        "hypotheses": hypotheses,
    }


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _worst(pvalues: Iterable[float | None]) -> float | None:
    """Intersection-union endpoint: the largest p, or None if any component is missing."""
    values = [value for value in pvalues]
    if any(value is None for value in values):
        return None
    return max(float(value) for value in values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, help="History summary.json")
    parser.add_argument("--scrambling", required=True, nargs=2, help="Whisper and Qwen summaries")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    scrambling = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.scrambling]
    result = confirmatory_family(history, scrambling)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
