from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.signal import resample_poly


EPSILON = 1e-12


def read_audio(path: str | Path, target_sample_rate: int) -> np.ndarray:
    """Read mono float32 audio, resampling when necessary.

    soundfile is imported lazily so array-only analysis does not require the
    audio I/O dependency. The inference environment must provide libsndfile.
    """
    try:
        import soundfile as sf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "reading FLAC/MUSDB audio requires soundfile; install the project runtime "
            "dependencies in the execution environment"
        ) from exc
    samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1, dtype=np.float32)
    if samples.ndim != 1:
        raise ValueError(f"expected mono or stereo audio: {path} shape={samples.shape}")
    if sample_rate != target_sample_rate:
        divisor = math.gcd(int(sample_rate), int(target_sample_rate))
        samples = resample_poly(
            samples,
            target_sample_rate // divisor,
            sample_rate // divisor,
        ).astype(np.float32)
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"audio contains NaN/Inf: {path}")
    return samples


def write_pcm16(path: str | Path, samples: np.ndarray, sample_rate: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f"write_pcm16 expects mono audio, got {audio.shape}")
    if not np.all(np.isfinite(audio)):
        raise ValueError("refusing to write NaN/Inf audio")
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(destination), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def read_pcm16(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 WAV: {path}")
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sample_rate


def rms(samples: np.ndarray) -> float:
    audio = np.asarray(samples, dtype=np.float64)
    return float(np.sqrt(np.mean(audio * audio) + EPSILON))


def peak(samples: np.ndarray) -> float:
    return float(np.max(np.abs(samples), initial=0.0))


def db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def vocal_gain_for_snr(speech: np.ndarray, vocal: np.ndarray, snr_db: float) -> float:
    """Gain for vocal v in y=s+g*v under SNR=20log10(RMS(s)/RMS(gv))."""
    return rms(speech) / (db_to_gain(snr_db) * rms(vocal))


def vad_onset(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float,
    hop_ms: float,
    threshold_db_below_peak: float,
    min_consecutive_frames: int,
    pre_roll_ms: float,
) -> tuple[int, dict[str, float]]:
    """Energy VAD used only to align t=0 to target-speech onset."""
    frame = max(1, round(frame_ms * sample_rate / 1000.0))
    hop = max(1, round(hop_ms * sample_rate / 1000.0))
    if len(samples) < frame:
        raise ValueError("audio is shorter than one VAD frame")
    starts = np.arange(0, len(samples) - frame + 1, hop, dtype=np.int64)
    energies = np.array([rms(samples[start : start + frame]) for start in starts])
    db = 20.0 * np.log10(np.maximum(energies, EPSILON))
    threshold = float(db.max() - threshold_db_below_peak)
    active = db >= threshold
    onset_frame = None
    run = max(1, int(min_consecutive_frames))
    for index in range(0, len(active) - run + 1):
        if bool(np.all(active[index : index + run])):
            onset_frame = index
            break
    if onset_frame is None:
        raise ValueError("energy VAD found no target-speech onset")
    onset = int(starts[onset_frame])
    onset = max(0, onset - round(pre_roll_ms * sample_rate / 1000.0))
    return onset, {
        "vad_peak_db": float(db.max()),
        "vad_threshold_db": threshold,
        "vad_onset_s": onset / sample_rate,
    }


def active_fraction(
    samples: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 50.0,
    threshold_db_below_peak: float = 30.0,
) -> float:
    frame = max(1, round(frame_ms * sample_rate / 1000.0))
    if len(samples) < frame:
        return 0.0
    usable = len(samples) // frame * frame
    frames = samples[:usable].reshape(-1, frame)
    energies = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + EPSILON)
    db = 20.0 * np.log10(np.maximum(energies, EPSILON))
    return float(np.mean(db >= db.max() - threshold_db_below_peak))


def required_windows_have_energy(
    samples: np.ndarray,
    sample_rate: int,
    windows_s: Sequence[Sequence[float]],
    min_db_below_clip_rms: float,
) -> tuple[bool, list[dict[str, float]]]:
    clip_rms = rms(samples)
    minimum = clip_rms * db_to_gain(-abs(min_db_below_clip_rms))
    checks: list[dict[str, float]] = []
    for start_s, end_s in windows_s:
        start = max(0, min(len(samples), round(float(start_s) * sample_rate)))
        end = max(0, min(len(samples), round(float(end_s) * sample_rate)))
        window_rms = rms(samples[start:end]) if end > start else 0.0
        checks.append(
            {
                "start_s": float(start_s),
                "end_s": float(end_s),
                "rms": window_rms,
                "db_relative_to_clip_rms": 20.0
                * math.log10(max(window_rms, EPSILON) / max(clip_rms, EPSILON)),
            }
        )
    return all(item["rms"] >= minimum for item in checks), checks


def select_active_crops(
    samples: np.ndarray,
    sample_rate: int,
    *,
    duration_s: float,
    count: int,
    hop_s: float,
    min_active_fraction: float,
    threshold_db_below_peak: float,
    required_activity_windows_s: Sequence[Sequence[float]] = (),
    component_window_min_db_below_clip_rms: float = 20.0,
    candidate_validator: Callable[[int, np.ndarray], bool] | None = None,
) -> list[tuple[int, np.ndarray, float]]:
    """Select high-activity non-overlapping vocal crops deterministically."""
    width = round(duration_s * sample_rate)
    hop = max(1, round(hop_s * sample_rate))
    candidates: list[tuple[float, int]] = []
    for start in range(0, max(0, len(samples) - width + 1), hop):
        crop = samples[start : start + width]
        fraction = active_fraction(
            crop,
            sample_rate,
            threshold_db_below_peak=threshold_db_below_peak,
        )
        if fraction >= min_active_fraction:
            if candidate_validator is not None and not candidate_validator(start, crop):
                continue
            windows_ok, _ = required_windows_have_energy(
                crop,
                sample_rate,
                required_activity_windows_s,
                component_window_min_db_below_clip_rms,
            )
            if windows_ok:
                candidates.append((rms(crop), start))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[tuple[int, np.ndarray, float]] = []
    for _, start in candidates:
        if any(abs(start - prior_start) < width for prior_start, _, _ in chosen):
            continue
        crop = samples[start : start + width].copy()
        fraction = active_fraction(
            crop,
            sample_rate,
            threshold_db_below_peak=threshold_db_below_peak,
        )
        chosen.append((start, crop, fraction))
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda item: item[0])


def attenuation_envelope(
    n_samples: int,
    sample_rate: int,
    intervals_s: Sequence[Sequence[float]],
    attenuation_db: float,
    ramp_ms: float,
) -> np.ndarray:
    """Return a gain envelope whose ramps are contained within each interval."""
    envelope = np.ones(n_samples, dtype=np.float32)
    low = db_to_gain(-abs(attenuation_db))
    ramp = max(0, round(ramp_ms * sample_rate / 1000.0))
    for interval in intervals_s:
        if len(interval) != 2:
            raise ValueError(f"interval must have [start, end], got {interval}")
        start = max(0, min(n_samples, round(float(interval[0]) * sample_rate)))
        end = max(0, min(n_samples, round(float(interval[1]) * sample_rate)))
        if end <= start:
            raise ValueError(f"empty intervention interval: {interval}")
        length = end - start
        local_ramp = min(ramp, length // 2)
        envelope[start:end] = low
        if start > 0 and local_ramp:
            phase = np.linspace(0.0, 1.0, local_ramp, endpoint=True)
            envelope[start : start + local_ramp] = 1.0 + (low - 1.0) * (
                0.5 - 0.5 * np.cos(np.pi * phase)
            )
        if end < n_samples and local_ramp:
            phase = np.linspace(0.0, 1.0, local_ramp, endpoint=True)
            envelope[end - local_ramp : end] = low + (1.0 - low) * (
                0.5 - 0.5 * np.cos(np.pi * phase)
            )
    return envelope


def piecewise_snr_envelope(
    speech: np.ndarray,
    vocal: np.ndarray,
    sample_rate: int,
    segments: Sequence[tuple[float, float, float]],
    ramp_ms: float,
) -> np.ndarray:
    """Create a vocal-gain envelope from (start_s, end_s, SNR_dB) segments."""
    n_samples = min(len(speech), len(vocal))
    envelope = np.empty(n_samples, dtype=np.float32)
    covered = np.zeros(n_samples, dtype=bool)
    gains: list[tuple[int, int, float]] = []
    for start_s, end_s, snr_db in segments:
        start = max(0, min(n_samples, round(start_s * sample_rate)))
        end = max(0, min(n_samples, round(end_s * sample_rate)))
        if end <= start:
            raise ValueError(f"invalid SNR segment {(start_s, end_s, snr_db)}")
        gain = vocal_gain_for_snr(speech, vocal, snr_db)
        envelope[start:end] = gain
        covered[start:end] = True
        gains.append((start, end, gain))
    if not bool(np.all(covered)):
        raise ValueError("piecewise SNR envelope does not cover the full clip")
    ramp = max(0, round(ramp_ms * sample_rate / 1000.0))
    for (_, left_end, left_gain), (right_start, _, right_gain) in zip(gains, gains[1:]):
        if left_end != right_start:
            raise ValueError("piecewise SNR segments must be contiguous")
        half = min(ramp // 2, left_end, n_samples - right_start)
        if not half:
            continue
        start, end = left_end - half, right_start + half
        phase = np.linspace(0.0, 1.0, end - start, endpoint=True)
        envelope[start:end] = left_gain + (right_gain - left_gain) * (
            0.5 - 0.5 * np.cos(np.pi * phase)
        )
    return envelope


def piecewise_gain_envelope(
    n_samples: int,
    sample_rate: int,
    segments: Sequence[tuple[float, float, float]],
    ramp_ms: float,
) -> np.ndarray:
    """Create a smooth envelope from explicit (start_s, end_s, gain) segments."""
    envelope = np.empty(n_samples, dtype=np.float32)
    covered = np.zeros(n_samples, dtype=bool)
    indexed: list[tuple[int, int, float]] = []
    for start_s, end_s, gain in segments:
        start = max(0, min(n_samples, round(start_s * sample_rate)))
        end = max(0, min(n_samples, round(end_s * sample_rate)))
        if end <= start:
            raise ValueError(f"invalid gain segment {(start_s, end_s, gain)}")
        envelope[start:end] = float(gain)
        covered[start:end] = True
        indexed.append((start, end, float(gain)))
    if not bool(np.all(covered)):
        raise ValueError("piecewise gain envelope does not cover the full clip")
    ramp = max(0, round(ramp_ms * sample_rate / 1000.0))
    for (_, left_end, left_gain), (right_start, _, right_gain) in zip(indexed, indexed[1:]):
        if left_end != right_start:
            raise ValueError("piecewise gain segments must be contiguous")
        half = min(ramp // 2, left_end, n_samples - right_start)
        if not half:
            continue
        start, end = left_end - half, right_start + half
        phase = np.linspace(0.0, 1.0, end - start, endpoint=True)
        envelope[start:end] = left_gain + (right_gain - left_gain) * (
            0.5 - 0.5 * np.cos(np.pi * phase)
        )
    return envelope


def compose_unscaled(
    speech: np.ndarray, vocal: np.ndarray, vocal_envelope: np.ndarray
) -> np.ndarray:
    n_samples = min(len(speech), len(vocal), len(vocal_envelope))
    return (
        speech[:n_samples].astype(np.float64)
        + vocal[:n_samples].astype(np.float64) * vocal_envelope[:n_samples]
    ).astype(np.float32)


def common_scale(mixtures: Iterable[np.ndarray], headroom_peak: float) -> float:
    maximum = max((peak(mixture) for mixture in mixtures), default=0.0)
    return 1.0 if maximum <= headroom_peak else float(headroom_peak / maximum)


def energy_dose(vocal: np.ndarray, baseline_gain: float, envelope: np.ndarray) -> float:
    """Removed vocal energy in mean-square-samples; comparable within a pair."""
    base = vocal.astype(np.float64) * float(baseline_gain)
    altered = vocal.astype(np.float64) * envelope.astype(np.float64)
    return float(np.sum(base * base - altered * altered))


def energy_dose_fraction(
    vocal: np.ndarray, baseline_gain: float, envelope: np.ndarray
) -> float:
    base = vocal.astype(np.float64) * float(baseline_gain)
    denominator = float(np.sum(base * base) + EPSILON)
    return energy_dose(vocal, baseline_gain, envelope) / denominator
