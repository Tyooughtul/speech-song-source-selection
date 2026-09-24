from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np


def two_way_cluster_bootstrap(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    track_key: str = "track_id",
    utterance_key: str = "speech_id",
    replicates: int = 10000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    """Pigeonhole bootstrap over crossed track and utterance clusters."""
    tracks = sorted({str(row[track_key]) for row in rows})
    utterances = sorted({str(row[utterance_key]) for row in rows})
    if not tracks or not utterances:
        return _empty_summary()
    track_index = {value: index for index, value in enumerate(tracks)}
    utterance_index = {value: index for index, value in enumerate(utterances)}
    cells: list[list[list[float]]] = [
        [[] for _ in utterances] for _ in tracks
    ]
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        cells[track_index[str(row[track_key])]][utterance_index[str(row[utterance_key])]].append(
            float(value)
        )
    matrix = np.full((len(tracks), len(utterances)), np.nan, dtype=np.float64)
    for track_i, values_by_utterance in enumerate(cells):
        for utterance_i, values in enumerate(values_by_utterance):
            if values:
                matrix[track_i, utterance_i] = float(np.mean(values))
    available = matrix[np.isfinite(matrix)]
    if not len(available):
        return _empty_summary()
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled_tracks = rng.integers(0, len(tracks), size=len(tracks))
        sampled_utterances = rng.integers(0, len(utterances), size=len(utterances))
        sampled = matrix[np.ix_(sampled_tracks, sampled_utterances)]
        draws[index] = np.nanmean(sampled)
    draws = draws[np.isfinite(draws)]
    return {
        "estimate": float(np.mean(available)),
        "ci95_low": float(np.quantile(draws, 0.025)) if len(draws) else None,
        "ci95_high": float(np.quantile(draws, 0.975)) if len(draws) else None,
        "n_tracks": len(tracks),
        "n_utterances": len(utterances),
        "n_observed_cells": int(np.isfinite(matrix).sum()),
    }


def exact_sign_flip_pvalue(values: list[float]) -> float | None:
    """Exact two-sided randomization p-value for cluster-aggregated effects."""
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return None
    if len(array) > 20:
        # Exact enumeration is used for the 13-track/15-utterance primary
        # confirmation set. Larger development sets would require >2^20 draws.
        return None
    observed = abs(float(array.mean()))
    extreme = 0
    total = 1 << len(array)
    for signs in product((-1.0, 1.0), repeat=len(array)):
        statistic = abs(float(np.mean(array * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return extreme / total


def sign_flip_pvalue(
    values: list[float],
    *,
    samples: int = 100_000,
    seed: int = 0,
) -> float | None:
    """Two-sided sign-flip p-value: exact for <=20 clusters, sampled otherwise.

    The sampled branch is a Monte Carlo randomization test over sign
    assignments, used for cluster counts above the exact-enumeration budget
    (e.g., the 26-track replication set).
    """
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not len(array):
        return None
    if len(array) <= 20:
        return exact_sign_flip_pvalue(values)
    observed = abs(float(array.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(int(samples), len(array)))
    statistics = np.abs((array * signs).mean(axis=1))
    extreme = int(np.count_nonzero(statistics >= observed - 1e-15))
    return (extreme + 1) / (int(samples) + 1)


def holm_adjust(pvalues: list[float | None]) -> list[float | None]:
    """Holm family-wise adjusted p-values, preserving input order."""
    output: list[float | None] = [None] * len(pvalues)
    valid = sorted(
        ((index, float(value)) for index, value in enumerate(pvalues) if value is not None),
        key=lambda item: item[1],
    )
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid):
        running = max(running, min(1.0, (count - rank) * value))
        output[index] = running
    return output


def _empty_summary() -> dict[str, float | int | None]:
    return {
        "estimate": None,
        "ci95_low": None,
        "ci95_high": None,
        "n_tracks": 0,
        "n_utterances": 0,
        "n_observed_cells": 0,
    }
