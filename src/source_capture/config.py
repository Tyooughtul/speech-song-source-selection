from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


PLACEHOLDER_PREFIX = "__SET_ME_"


def load_config(path: str | Path, *, require_paths: bool = True) -> dict[str, Any]:
    """Load a configuration file and resolve relative paths against its directory.

    Placeholder checking is left to the stage that consumes the values, through
    :func:`require_filled`, so that stimulus preparation is not blocked by model
    paths it never reads.
    """
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(config_path)
    config["_config_dir"] = str(config_path.parent)
    _resolve_known_paths(config)
    if require_paths:
        placeholders = find_placeholders(config)
        if placeholders:
            rendered = "\n".join(f"  - {key}: {value}" for key, value in placeholders)
            raise ValueError(
                "configuration still contains placeholders:\n"
                f"{rendered}\nCopy the placeholder config and fill these values."
            )
    return config


def _resolve_known_paths(config: dict[str, Any]) -> None:
    base = Path(config["_config_dir"])
    for section, key in (
        ("project", "output_root"),
        ("datasets", "librispeech_test_clean"),
        ("datasets", "musdb18hq_root"),
        ("datasets", "lyrics_manifest"),
        ("datasets", "speech_alignment_manifest"),
        ("models.whisper", "model_path"),
        ("models.ctc", "word_alignment_manifest"),
        ("models.ctc", "model_path"),
    ):
        node: Any = config
        parts = section.split(".")
        for part in parts:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        value = node.get(key) if isinstance(node, dict) else None
        if not isinstance(value, str) or value.startswith(PLACEHOLDER_PREFIX):
            continue
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            expanded = (base / expanded).resolve()
        node[key] = str(expanded)
    for model in config.get("models", {}).values():
        if not isinstance(model, dict):
            continue
        for key in ("model_path", "word_alignment_manifest"):
            value = model.get(key)
            if not isinstance(value, str) or value.startswith(PLACEHOLDER_PREFIX):
                continue
            expanded = Path(value).expanduser()
            if not expanded.is_absolute():
                expanded = (base / expanded).resolve()
            model[key] = str(expanded)


def find_placeholders(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(find_placeholders(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_placeholders(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX):
        found.append((prefix, value))
    return found


def require_filled(config: dict[str, Any], *keys: str) -> None:
    """Raise if a placeholder survives inside the given configuration subtrees.

    Each key is a dotted path, for example ``models.whisper`` or ``datasets``.
    Stage entry points call this for the subtrees they actually consume, so that
    generating stimuli does not demand a model checkpoint it never loads.
    """
    found: list[tuple[str, str]] = []
    for key in keys:
        found.extend(find_placeholders(_node(config, key), key))
    if found:
        rendered = "\n".join(f"  - {name}: {value}" for name, value in sorted(set(found)))
        raise ValueError(
            "configuration still contains placeholders:\n"
            f"{rendered}\nFill these values before running this stage."
        )


def _node(config: dict[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"missing configuration section: {dotted}")
        node = node[part]
    return node


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["project"]["output_root"])


def experiment_dir(config: dict[str, Any], experiment: str) -> Path:
    if experiment not in {"onset_position", "temporal_history"}:
        raise ValueError(f"unknown experiment: {experiment}")
    return output_root(config) / experiment
