"""Shared output-level summaries used across paper experiments."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any


OUTPUT_TYPES = ("pure_speech", "pure_lyric", "mixed", "no_grounded")


def output_type(row: dict[str, Any]) -> str:
    """Classify an output using tokens uniquely grounded to either source."""
    if row.get("no_grounded_output"):
        return "no_grounded"
    if row.get("lir") == 0.0:
        return "pure_speech"
    if row.get("lir") == 1.0:
        return "pure_lyric"
    return "mixed"


def source_transitions(row: dict[str, Any]) -> int | None:
    """Count switches between speech- and lyric-attributed output tokens."""
    attribution = row.get("token_attribution")
    if not attribution:
        return None
    sequence = [
        item["source"]
        for item in attribution
        if item.get("source") in {"speech", "lyric"}
    ]
    return sum(left != right for left, right in zip(sequence, sequence[1:]))


def output_selection_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return output-type rates and transition count among mixed outputs."""
    counts: Counter[str] = Counter()
    transitions: list[int] = []
    for row in rows:
        category = output_type(row)
        counts[category] += 1
        if category == "mixed":
            value = source_transitions(row)
            if value is not None:
                transitions.append(value)
    total = len(rows)
    return {
        "n": total,
        "output_type_rate": {
            category: counts[category] / total if total else None
            for category in OUTPUT_TYPES
        },
        "mean_source_transitions_mixed": mean(transitions) if transitions else None,
    }
