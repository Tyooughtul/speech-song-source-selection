"""Single command-line interface for every experiment in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_config, require_filled
from .core.scoring import score_run
from .dataset import prepare_pairs
from .experiments import front_end, full_mix, history_replication, scrambling
from .experiments.position_history_analysis import analyze_experiment
from .experiments.position_history import generate_experiment
from .inference import run_model


PAPER_EXPERIMENTS = ("position", "full-mix", "history", "scrambling", "front-end")

# Configuration subtrees each stage actually reads. Checking per stage keeps
# stimulus preparation from demanding a model checkpoint it never loads.
MODEL_STAGE_KEYS = ("project", "models")
MODULE_STAGE_KEYS = {
    "prepare": ("project", "audio", "scoring"),
    "enhance": ("project", "frontend"),
    "transcribe": ("project", "audio", "scoring", "models"),
    "analyze": ("project", "audio", "scoring"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="source-capture",
        description="Reproduce the five speech-song source-selection experiments.",
    )
    parser.add_argument("experiment", choices=("dataset",) + PAPER_EXPERIMENTS)
    parser.add_argument(
        "stage",
        choices=("prepare", "generate", "enhance", "transcribe", "align", "score", "analyze"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("whisper", "qwen3", "ctc"), default="whisper")
    parser.add_argument("--split", choices=("development", "evaluation", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment, stage = args.experiment, args.stage

    if experiment == "dataset":
        _require_stage(stage, "prepare")
        config = load_config(args.config, require_paths=False)
        require_filled(config, "project", "datasets", "selection", "audio", "scoring")
        print(prepare_pairs(config))
        return 0

    if experiment in {"position", "history"}:
        if experiment == "history" and _is_replication_config(args.config):
            return _run_history_replication(args)
        experiment_key = "onset_position" if experiment == "position" else "temporal_history"
        config = load_config(args.config, require_paths=False)
        if stage == "generate":
            require_filled(config, "project", "audio", "experiments", "scoring")
            print(generate_experiment(config, experiment_key))
        elif stage == "transcribe":
            require_filled(config, "project", f"models.{args.model}")
            print(run_model(config, experiment_key, args.model, overwrite=args.overwrite, split=args.split))
        elif stage == "score":
            require_filled(config, "project", "scoring", f"models.{args.model}")
            print(score_run(config, experiment_key, args.model))
        elif stage == "analyze":
            require_filled(config, "project", "scoring")
            for path in analyze_experiment(config, experiment_key):
                print(path)
        else:
            _require_stage(stage, "generate", "transcribe", "score", "analyze")
        return 0

    module = {
        "full-mix": full_mix,
        "scrambling": scrambling,
        "front-end": front_end,
    }[experiment]
    config = module.load_config(args.config)
    _check_module_stage(experiment, stage, config, args.model)
    if stage == "prepare":
        print(module.prepare(config))
    elif stage == "enhance" and experiment == "front-end":
        print(module.enhance(config, limit_pairs=args.limit))
    elif stage == "transcribe":
        print(module.run(config, model_name=args.model, limit=args.limit))
    elif stage == "analyze":
        print(module.analyze(config, model_name=args.model))
    else:
        allowed = ("prepare", "enhance", "transcribe", "analyze") if experiment == "front-end" else ("prepare", "transcribe", "analyze")
        _require_stage(stage, *allowed)
    return 0


def _check_module_stage(experiment: str, stage: str, config: dict, model: str) -> None:
    if stage not in MODULE_STAGE_KEYS:
        return
    keys = list(MODULE_STAGE_KEYS[stage])
    if stage == "transcribe":
        keys[keys.index("models")] = f"models.{model}"
    if experiment in {"scrambling", "front-end"}:
        keys.append("selection")
    if experiment == "front-end":
        keys.append("frontend")
    require_filled(config, *keys)


def _run_history_replication(args: argparse.Namespace) -> int:
    config = history_replication.load_config(args.config)
    stage = args.stage
    if stage == "prepare":
        require_filled(config, "project", "experiment", "audio", "scoring")
        print(history_replication.prepare(config))
    elif stage == "transcribe":
        require_filled(config, "project", f"models.{args.model}")
        print(history_replication.run(config, model_name=args.model, limit=args.limit))
    elif stage == "align":
        require_filled(config, "project", "models.ctc")
        print(history_replication.align(config, model_name=args.model))
    elif stage == "analyze":
        require_filled(config, "project", "scoring")
        print(history_replication.analyze(config, model_name=args.model))
    else:
        _require_stage(stage, "prepare", "transcribe", "align", "analyze")
    return 0


def _is_replication_config(path: str) -> bool:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return isinstance(value, dict) and "experiment" in value and "experiments" not in value


def _require_stage(actual: str, *allowed: str) -> None:
    if actual not in allowed:
        raise SystemExit(f"stage {actual!r} is not valid here; choose one of: {', '.join(allowed)}")
