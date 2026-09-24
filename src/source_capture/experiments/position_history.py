from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.audio import (
    attenuation_envelope,
    common_scale,
    compose_unscaled,
    energy_dose,
    energy_dose_fraction,
    peak,
    piecewise_gain_envelope,
    piecewise_snr_envelope,
    read_pcm16,
    rms,
    vocal_gain_for_snr,
    write_pcm16,
)
from ..config import experiment_dir
from ..core.io import read_jsonl, sha256_file, stable_id, write_jsonl


Condition = dict[str, Any]


def generate_experiment(config: dict[str, Any], experiment: str) -> Path:
    if experiment not in {"onset_position", "temporal_history"}:
        raise ValueError(f"unknown experiment: {experiment}")
    pairs_path = Path(config["project"]["output_root"]) / "shared" / "pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(f"prepare pairs first: {pairs_path}")
    destination = experiment_dir(config, experiment)
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_generation(destination, audio_dir)
    builder: Callable[..., list[Condition]] = {
        "onset_position": _onset_position_conditions,
        "temporal_history": _temporal_history_conditions,
    }[experiment]

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for pair in read_jsonl(pairs_path):
        speech, speech_sr = read_pcm16(pair["speech_path"])
        vocal, vocal_sr = read_pcm16(pair["vocal_path"])
        sample_rate = int(config["audio"]["sample_rate"])
        if speech_sr != sample_rate or vocal_sr != sample_rate:
            raise ValueError(f"prepared audio sample-rate mismatch for {pair['pair_id']}")
        n_samples = min(len(speech), len(vocal))
        speech, vocal = speech[:n_samples], vocal[:n_samples]
        conditions = builder(config, speech, vocal)
        mixtures = [compose_unscaled(speech, vocal, row["vocal_envelope"]) for row in conditions]
        scale_groups: dict[str, list[np.ndarray]] = {}
        for condition, mixture in zip(conditions, mixtures):
            scale_groups.setdefault(condition["scale_group"], []).append(mixture)
        scales = {
            group: common_scale(group_mixtures, float(config["audio"]["headroom_peak"]))
            for group, group_mixtures in scale_groups.items()
        }
        for condition, unscaled in zip(conditions, mixtures):
            scale_group = condition["scale_group"]
            scale = scales[scale_group]
            family_id = stable_id(experiment, pair["pair_id"], scale_group)
            sample_id = stable_id(experiment, pair["pair_id"], condition["condition_id"])
            output = audio_dir / f"{sample_id}.wav"
            final = unscaled * scale
            write_pcm16(output, final, sample_rate)
            envelope = condition.pop("vocal_envelope")
            row = {
                "sample_id": sample_id,
                "family_id": family_id,
                "experiment": experiment,
                "audio_path": str(output.resolve()),
                "audio_sha256": sha256_file(output),
                "common_scale": scale,
                **pair,
                **condition,
            }
            rows.append(row)
            checks.append(
                {
                    "sample_id": sample_id,
                    "pair_id": pair["pair_id"],
                    "condition_id": condition["condition_id"],
                    "unscaled_peak": peak(unscaled),
                    "final_peak": peak(final),
                    "final_speech_rms": rms(speech * scale),
                    "final_vocal_rms": rms(vocal * envelope * scale),
                    "common_scale": scale,
                    "clipped": bool(peak(final) > 1.0 + 1e-7),
                }
            )
    manifest = destination / "manifest.jsonl"
    write_jsonl(manifest, rows)
    (destination / "audio_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _clear_previous_generation(destination: Path, audio_dir: Path) -> None:
    """Regeneration invalidates downstream outputs; remove only scoped artifacts."""
    for wav in audio_dir.glob("*.wav"):
        wav.unlink()
    for name in ("manifest.jsonl", "audio_checks.json", "summary.json", "summary.csv"):
        path = destination / name
        if path.exists():
            path.unlink()
    for child_dir in (destination / "runs", destination / "scores"):
        if not child_dir.exists():
            continue
        for artifact in child_dir.glob("*.jsonl"):
            artifact.unlink()


def _onset_position_conditions(
    config: dict[str, Any], speech: np.ndarray, vocal: np.ndarray
) -> list[Condition]:
    params = config["experiments"]["onset_position"]
    sample_rate = int(config["audio"]["sample_rate"])
    clip_s = float(config["audio"]["clip_duration_s"])
    ramp_ms = float(config["audio"]["intervention_ramp_ms"])
    duration = float(params["intervention_duration_s"])
    attenuation = float(params["attenuation_db"])
    positions = {
        "baseline": [],
        "onset": [[0.0, duration]],
        "middle": [[(clip_s - duration) / 2.0, (clip_s + duration) / 2.0]],
        "offset": [[clip_s - duration, clip_s]],
        "distributed": params["distributed_intervals_s"],
        "full": [[0.0, clip_s]],
    }
    conditions: list[Condition] = []
    for snr_db in params["snr_db"]:
        baseline_gain = vocal_gain_for_snr(speech, vocal, float(snr_db))
        for position, intervals in positions.items():
            if position == "baseline":
                envelope = np.full(len(speech), baseline_gain, dtype=np.float32)
                applied_attenuation = 0.0
            else:
                local = attenuation_envelope(
                    len(speech), sample_rate, intervals, attenuation, ramp_ms
                )
                envelope = local * baseline_gain
                applied_attenuation = attenuation
            conditions.append(
                {
                    "condition_id": f"snr{_num_tag(snr_db)}_{position}",
                    "scale_group": f"snr{_num_tag(snr_db)}",
                    "condition": position,
                    "snr_db": float(snr_db),
                    "attenuation_db": applied_attenuation,
                    "intervention_intervals_s": intervals,
                    "nominal_intervention_duration_s": sum(
                        float(end) - float(start) for start, end in intervals
                    ),
                    "energy_dose": energy_dose(vocal, baseline_gain, envelope),
                    "energy_dose_fraction": energy_dose_fraction(
                        vocal, baseline_gain, envelope
                    ),
                    "eligible_models": ["whisper", "qwen3", "ctc"],
                    "vocal_envelope": envelope,
                }
            )
    return conditions


def _temporal_history_conditions(
    config: dict[str, Any], speech: np.ndarray, vocal: np.ndarray
) -> list[Condition]:
    params = config["experiments"]["temporal_history"]
    sample_rate = int(config["audio"]["sample_rate"])
    clip_s = float(config["audio"]["clip_duration_s"])
    ramp_ms = float(config["audio"]["intervention_ramp_ms"])
    speech_snr = float(params["speech_dominant_snr_db"])
    lyric_snr = float(params["lyric_dominant_snr_db"])
    equal_snr = float(params["equal_snr_db"])
    conditions: list[Condition] = []

    def add(
        condition_id: str,
        condition: str,
        segments: list[tuple[float, float, float]],
        switch_s: float | None,
        comparator: str | None,
        comparison_direction: str | None,
    ) -> None:
        envelope = piecewise_snr_envelope(speech, vocal, sample_rate, segments, ramp_ms)
        add_envelope(
            condition_id,
            condition,
            envelope,
            segments,
            switch_s,
            comparator,
            comparison_direction,
        )

    def add_envelope(
        condition_id: str,
        condition: str,
        envelope: np.ndarray,
        segments: list[tuple[float, float, float]],
        switch_s: float | None,
        comparator: str | None,
        comparison_direction: str | None,
    ) -> None:
        ramp_half_s = ramp_ms / 2000.0 if switch_s is not None else 0.0
        evaluation_start = None if switch_s is None else switch_s + ramp_half_s
        conditions.append(
            {
                "condition_id": condition_id,
                "scale_group": "temporal_history_all_conditions",
                "condition": condition,
                "snr_segments": [
                    {"start_s": start, "end_s": end, "snr_db": snr}
                    for start, end, snr in segments
                ],
                "switch_s": switch_s,
                "evaluation_start_s": evaluation_start,
                "evaluation_end_s": (
                    None
                    if evaluation_start is None
                    else min(
                        clip_s,
                        evaluation_start + float(params["post_switch_evaluation_s"]),
                    )
                ),
                "static_comparator_condition": comparator,
                "history_effect_direction": comparison_direction,
                "primary_condition": bool(
                    switch_s is not None
                    and switch_s == float(params["primary_history_duration_s"])
                ),
                "vocal_energy_pre_common_scale": float(
                    np.sum((vocal.astype(np.float64) * envelope.astype(np.float64)) ** 2)
                ),
                "eligible_models": ["whisper", "qwen3", "ctc"],
                "vocal_envelope": envelope,
            }
        )

    add(
        "static_speech",
        "static_speech",
        [(0.0, clip_s, speech_snr)],
        None,
        None,
        None,
    )
    add(
        "static_lyric",
        "static_lyric",
        [(0.0, clip_s, lyric_snr)],
        None,
        None,
        None,
    )
    add(
        "static_equal",
        "static_equal",
        [(0.0, clip_s, equal_snr)],
        None,
        None,
        None,
    )
    for history_s in params["history_durations_s"]:
        tau = float(history_s)
        evaluation_s = float(params["post_switch_evaluation_s"])
        evaluation_end = min(clip_s, tau + ramp_ms / 2000.0 + evaluation_s)
        pre_speech_gain = vocal_gain_for_snr(
            speech[: round(tau * sample_rate)],
            vocal[: round(tau * sample_rate)],
            speech_snr,
        )
        pre_lyric_gain = vocal_gain_for_snr(
            speech[: round(tau * sample_rate)],
            vocal[: round(tau * sample_rate)],
            lyric_snr,
        )
        post_slice = slice(round((tau + ramp_ms / 2000.0) * sample_rate), round(evaluation_end * sample_rate))
        post_lyric_gain = vocal_gain_for_snr(
            speech[post_slice], vocal[post_slice], lyric_snr
        )
        post_speech_gain = vocal_gain_for_snr(
            speech[post_slice], vocal[post_slice], speech_snr
        )
        s_to_l_envelope = piecewise_gain_envelope(
            len(speech),
            sample_rate,
            [(0.0, tau, pre_speech_gain), (tau, clip_s, post_lyric_gain)],
            ramp_ms,
        )
        l_to_s_envelope = piecewise_gain_envelope(
            len(speech),
            sample_rate,
            [(0.0, tau, pre_lyric_gain), (tau, clip_s, post_speech_gain)],
            ramp_ms,
        )
        lyric_comparator_id = f"matched_lyric_t{_num_tag(tau)}"
        speech_comparator_id = f"matched_speech_t{_num_tag(tau)}"
        add_envelope(
            f"s_to_l_t{_num_tag(tau)}",
            "s_to_l",
            s_to_l_envelope,
            [(0.0, tau, speech_snr), (tau, clip_s, lyric_snr)],
            tau,
            lyric_comparator_id,
            "negative",
        )
        add_envelope(
            f"l_to_s_t{_num_tag(tau)}",
            "l_to_s",
            l_to_s_envelope,
            [(0.0, tau, lyric_snr), (tau, clip_s, speech_snr)],
            tau,
            speech_comparator_id,
            "positive",
        )
        add_envelope(
            lyric_comparator_id,
            "matched_static_lyric",
            np.full(len(speech), post_lyric_gain, dtype=np.float32),
            [(0.0, clip_s, lyric_snr)],
            None,
            None,
            None,
        )
        add_envelope(
            speech_comparator_id,
            "matched_static_speech",
            np.full(len(speech), post_speech_gain, dtype=np.float32),
            [(0.0, clip_s, speech_snr)],
            None,
            None,
            None,
        )
    symmetric = float(params["symmetric_switch_s"])
    symmetric_a = piecewise_snr_envelope(
        speech,
        vocal,
        sample_rate,
        [(0.0, symmetric, speech_snr), (symmetric, clip_s, lyric_snr)],
        ramp_ms,
    )
    symmetric_b = piecewise_snr_envelope(
        speech,
        vocal,
        sample_rate,
        [(0.0, symmetric, lyric_snr), (symmetric, clip_s, speech_snr)],
        ramp_ms,
    )
    energy_a = float(np.sum((vocal.astype(np.float64) * symmetric_a) ** 2))
    energy_b = float(np.sum((vocal.astype(np.float64) * symmetric_b) ** 2))
    target_energy = (energy_a * energy_b) ** 0.5
    symmetric_a *= float(np.sqrt(target_energy / max(energy_a, 1e-12)))
    symmetric_b *= float(np.sqrt(target_energy / max(energy_b, 1e-12)))
    add_envelope(
        "s5_to_l5",
        "symmetric_s_to_l",
        symmetric_a,
        [(0.0, symmetric, speech_snr), (symmetric, clip_s, lyric_snr)],
        symmetric,
        "l5_to_s5",
        None,
    )
    add_envelope(
        "l5_to_s5",
        "symmetric_l_to_s",
        symmetric_b,
        [(0.0, symmetric, lyric_snr), (symmetric, clip_s, speech_snr)],
        symmetric,
        "s5_to_l5",
        None,
    )
    return conditions


def _num_tag(value: float) -> str:
    numeric = float(value)
    sign = "p" if numeric >= 0 else "m"
    body = f"{abs(numeric):g}".replace(".", "p")
    return f"{sign}{body}"
