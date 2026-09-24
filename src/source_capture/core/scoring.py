from __future__ import annotations

import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import experiment_dir
from .io import load_json_or_jsonl, read_jsonl, write_jsonl


def normalize_words(text: str, config: dict[str, Any]) -> list[str]:
    value = text or ""
    if config.get("lowercase", True):
        value = value.lower()
    if config.get("strip_punctuation", True):
        value = value.translate(str.maketrans("", "", string.punctuation))
    return [word for word in re.split(r"\s+", value.strip()) if word]


def attribute_tokens(
    hypothesis: str,
    speech_reference: str,
    lyric_reference: str,
    scoring_config: dict[str, Any],
) -> dict[str, Any]:
    """Conservative dual-reference attribution using exclusive token capacity.

    Common words present in both references are marked ambiguous and excluded
    from LIR/SCS. Repeated matches cannot exceed their reference multiplicity.
    """
    hypothesis_words = normalize_words(hypothesis, scoring_config)
    speech_words = normalize_words(speech_reference, scoring_config)
    lyric_words = normalize_words(lyric_reference, scoring_config)
    speech_capacity = Counter(speech_words)
    lyric_capacity = Counter(lyric_words)
    common_vocabulary = set(speech_capacity) & set(lyric_capacity)
    n_speech_exclusive_reference = sum(
        count for word, count in speech_capacity.items() if word not in common_vocabulary
    )
    n_lyric_exclusive_reference = sum(
        count for word, count in lyric_capacity.items() if word not in common_vocabulary
    )
    n_speech_shared_reference = len(speech_words) - n_speech_exclusive_reference
    n_lyric_shared_reference = len(lyric_words) - n_lyric_exclusive_reference
    labels: list[dict[str, str]] = []
    counts = Counter()
    for word in hypothesis_words:
        if word in common_vocabulary:
            label = "ambiguous"
        elif speech_capacity[word] > 0:
            label = "speech"
            speech_capacity[word] -= 1
        elif lyric_capacity[word] > 0:
            label = "lyric"
            lyric_capacity[word] -= 1
        else:
            label = "other"
        counts[label] += 1
        labels.append({"word": word, "source": label})
    grounded = counts["speech"] + counts["lyric"]
    lir = counts["lyric"] / grounded if grounded else None
    scs = (counts["lyric"] - counts["speech"]) / grounded if grounded else None
    speech_exclusive_recall = (
        counts["speech"] / n_speech_exclusive_reference
        if n_speech_exclusive_reference
        else None
    )
    lyric_exclusive_recall = (
        counts["lyric"] / n_lyric_exclusive_reference
        if n_lyric_exclusive_reference
        else None
    )
    tsr = counts["speech"] / len(speech_words) if speech_words else None
    return {
        "normalized_hypothesis": hypothesis_words,
        "token_attribution": labels,
        "n_speech": counts["speech"],
        "n_lyric": counts["lyric"],
        "n_ambiguous": counts["ambiguous"],
        "n_other": counts["other"],
        "n_grounded": grounded,
        "n_speech_reference": len(speech_words),
        "n_lyric_reference": len(lyric_words),
        "n_speech_exclusive_reference": n_speech_exclusive_reference,
        "n_lyric_exclusive_reference": n_lyric_exclusive_reference,
        "n_speech_shared_reference": n_speech_shared_reference,
        "n_lyric_shared_reference": n_lyric_shared_reference,
        "speech_shared_reference_rate": (
            n_speech_shared_reference / len(speech_words) if speech_words else None
        ),
        "lyric_shared_reference_rate": (
            n_lyric_shared_reference / len(lyric_words) if lyric_words else None
        ),
        "speech_exclusive_recall": speech_exclusive_recall,
        "lyric_exclusive_recall": lyric_exclusive_recall,
        "lir": lir,
        "tsr": tsr,
        "scs": scs,
        "no_grounded_output": grounded == 0,
    }


def score_run(config: dict[str, Any], experiment: str, model_name: str) -> Path:
    source = experiment_dir(config, experiment) / "runs" / f"{model_name}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"run model first: {source}")
    destination = experiment_dir(config, experiment) / "scores" / f"{model_name}.jsonl"
    require_lyrics = bool(config["scoring"].get("require_lyric_reference", True))
    require_crop_lyrics = bool(
        config["scoring"].get("require_crop_aligned_lyrics", True)
    )
    output: list[dict[str, Any]] = []
    external_alignments = _load_output_alignments(config, model_name)
    prefer_external = bool(
        config["models"].get(model_name, {}).get(
            "prefer_external_word_alignment", False
        )
    )
    require_external = bool(
        config["models"].get(model_name, {}).get(
            "require_external_word_alignment", False
        )
    )
    if require_external and not external_alignments:
        raise ValueError(
            f"models.{model_name}.require_external_word_alignment is set but "
            f"models.{model_name}.word_alignment_manifest is missing or unreadable. "
            "Point it at the alignment manifest, or turn the requirement off to use "
            "model-native word timestamps."
        )
    missing = 0
    for row in read_jsonl(source):
        if row["sample_id"] in external_alignments and (
            prefer_external or not row.get("words")
        ):
            if prefer_external and row.get("words"):
                row["native_words"] = row["words"]
            row["words"] = external_alignments[row["sample_id"]]
            row["word_alignment_source"] = (
                "external_preferred" if prefer_external else "external_fallback"
            )
        elif prefer_external and require_external:
            if row.get("words"):
                row["native_words"] = row["words"]
            row["words"] = None
            row["word_alignment_source"] = "external_missing"
        lyrics = str(row.get("lyric_reference") or "")
        if not lyrics:
            missing += 1
            score = empty_score("missing lyric reference")
        elif require_crop_lyrics and row.get("lyric_reference_scope") != "crop":
            missing += 1
            score = empty_score("lyric reference is not crop-aligned")
        elif row.get("error"):
            score = empty_score("inference error")
        else:
            score = attribute_tokens(
                str(row.get("hyp", "")),
                str(row.get("speech_reference", "")),
                lyrics,
                config["scoring"],
            )
            score["score_error"] = None
        output.append({**row, **score})
    if require_lyrics and missing:
        raise RuntimeError(
            f"{missing}/{len(output)} rows lack a usable crop-aligned lyric reference; "
            "confirmation scoring aborted"
        )
    write_jsonl(destination, output)
    return destination


def _load_output_alignments(
    config: dict[str, Any], model_name: str
) -> dict[str, list[dict[str, Any]]]:
    path = config["models"].get(model_name, {}).get("word_alignment_manifest")
    if not path or str(path).startswith("__SET_ME_"):
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for row in load_json_or_jsonl(path):
        if "sample_id" not in row or "words" not in row:
            raise ValueError("output alignment records need sample_id and words")
        if row.get("alignment_error") or row.get("words") is None:
            continue
        output[str(row["sample_id"])] = list(row["words"])
    return output


def empty_score(reason: str) -> dict[str, Any]:
    return {
        "normalized_hypothesis": [],
        "token_attribution": [],
        "n_speech": 0,
        "n_lyric": 0,
        "n_ambiguous": 0,
        "n_other": 0,
        "n_grounded": 0,
        "n_speech_reference": 0,
        "n_lyric_reference": 0,
        "n_speech_exclusive_reference": 0,
        "n_lyric_exclusive_reference": 0,
        "n_speech_shared_reference": 0,
        "n_lyric_shared_reference": 0,
        "speech_shared_reference_rate": None,
        "lyric_shared_reference_rate": None,
        "speech_exclusive_recall": None,
        "lyric_exclusive_recall": None,
        "lir": None,
        "tsr": None,
        "scs": None,
        "no_grounded_output": True,
        "score_error": reason,
    }


def score_time_window(
    hypothesis_words: list[dict[str, Any]] | None,
    start_s: float,
    end_s: float,
    speech_reference: str,
    lyric_reference: str,
    scoring_config: dict[str, Any],
    *,
    speech_words: list[dict[str, Any]] | None = None,
    lyric_words: list[dict[str, Any]] | None = None,
    boundary_exclusion_s: float = 0.0,
) -> dict[str, Any] | None:
    if hypothesis_words is None:
        return None
    selected = _midpoint_window_words(
        hypothesis_words, start_s, end_s, boundary_exclusion_s
    )
    if speech_words is not None:
        speech_reference = " ".join(
            _midpoint_window_words(
                speech_words, start_s, end_s, boundary_exclusion_s
            )
        )
    if lyric_words is not None:
        lyric_reference = " ".join(
            _midpoint_window_words(
                lyric_words, start_s, end_s, boundary_exclusion_s
            )
        )
    return attribute_tokens(
        " ".join(selected), speech_reference, lyric_reference, scoring_config
    )


def _midpoint_window_words(
    words: list[dict[str, Any]],
    start_s: float,
    end_s: float,
    boundary_exclusion_s: float,
) -> list[str]:
    margin = max(0.0, float(boundary_exclusion_s))
    inner_start = start_s + margin
    inner_end = end_s - margin
    if inner_end <= inner_start:
        raise ValueError("boundary exclusion removes the entire scoring window")
    selected: list[str] = []
    for item in words:
        word = str(item.get("word", "")).strip()
        if not word:
            continue
        start = float(item.get("start", 1e9))
        end = float(item.get("end", -1e9))
        midpoint = 0.5 * (start + end)
        if inner_start <= midpoint < inner_end:
            selected.append(word)
    return selected
