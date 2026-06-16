"""Configuration loading and precedence helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .errors import TranscribeError


CONFIG_KEYS = {
    "language",
    "model",
    "model_path",
    "model_arch",
    "threads",
}


def default_config_path() -> Path:
    return Path.home() / ".config" / "scriba" / "config.toml"


def load_config(path: Path | None = None) -> dict[str, str]:
    config_path = path or default_config_path()
    if not config_path.exists():
        return {}

    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise TranscribeError(f"invalid config file {config_path}: {exc}") from exc
    except OSError as exc:
        raise TranscribeError(f"cannot read config file {config_path}: {exc}") from exc

    raw_section: Any = data.get("scriba", data)
    if not isinstance(raw_section, dict):
        raise TranscribeError(f"invalid config file {config_path}: [scriba] must be a table")

    result: dict[str, str] = {}
    for key in CONFIG_KEYS:
        if key not in raw_section:
            continue
        value = raw_section[key]
        if value is None:
            continue
        if not isinstance(value, (str, int)):
            raise TranscribeError(
                f"invalid config file {config_path}: {key} must be a string or integer"
            )
        result[key] = str(value)
    return result


def resolve_setting(
    cli_value: str | None,
    env_name: str,
    config: dict[str, str],
    config_key: str,
    default: str | None = None,
) -> str | None:
    if cli_value not in (None, ""):
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value not in (None, ""):
        return env_value
    config_value = config.get(config_key)
    if config_value not in (None, ""):
        return config_value
    return default
