from __future__ import annotations

import argparse
import copy
import inspect
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime for clearer CLI errors.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read training config files. Install pyyaml in the active env.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | os.PathLike[str]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write config snapshots. Install pyyaml in the active env.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." not in value and "e" not in lowered:
            return int(value)
        return float(value)
    except ValueError:
        return value


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value format, got: {item}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(updated, key, parse_scalar(raw_value))
    return updated


def resolve_path(path: str | None, base_dir: Path = REPO_ROOT) -> str | None:
    if path is None:
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(path)))
    p = Path(expanded)
    if not p.is_absolute():
        p = base_dir / p
    return str(p)


def config_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config values with dotted keys, e.g. --set training.learning_rate=1e-5.",
    )
    return parser


def load_config_with_overrides(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    return apply_overrides(load_yaml(config_path), overrides)


def prepare_run_dir(config: dict[str, Any]) -> Path:
    output_dir = resolve_path(config["output"]["output_dir"])
    run_dir = Path(output_dir)
    for child in ("checkpoints", "logs", "metrics", "configs", "generated_samples"):
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "configs" / "resolved_config.yaml")
    return run_dir


class TeeStream:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self) -> int:
        for stream in self.streams:
            fileno = getattr(stream, "fileno", None)
            if fileno is None:
                continue
            try:
                return fileno()
            except Exception:
                continue
        raise OSError("No wrapped stream exposes fileno().")

    @property
    def encoding(self) -> str:
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self) -> str:
        return getattr(self.streams[0], "errors", "replace")


def setup_run_logging(run_dir: Path, name: str) -> None:
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_file = open(logs_dir / f"{name}_stdout.log", "a", encoding="utf-8", buffering=1)
    stderr_file = open(logs_dir / f"{name}_stderr.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, stdout_file)
    sys.stderr = TeeStream(sys.__stderr__, stderr_file)


def accepted_kwargs(cls_or_fn: Any, values: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls_or_fn).parameters
    return {key: value for key, value in values.items() if key in params}


def apply_common_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    mapping = {
        "model": "model.model_name_or_path",
        "data": "data.train_file",
        "output_dir": "output.output_dir",
        "max_samples": "data.max_samples",
        "max_steps": "training.max_steps",
    }
    for attr, dotted_key in mapping.items():
        value = getattr(args, attr, None)
        if value is not None:
            set_by_dotted_key(updated, dotted_key, value)
    return updated
