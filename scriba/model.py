"""Moonshine model resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import TranscribeError


ARCH_NAMES = {
    "tiny",
    "base",
    "tiny-streaming",
    "base-streaming",
    "small-streaming",
    "medium-streaming",
}


def normalize_arch_name(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def parse_model_arch(raw_value: str | int | Any):
    """Parse a model architecture after importing Moonshine lazily."""
    try:
        from moonshine_voice import ModelArch, string_to_model_arch
    except ImportError as exc:
        raise TranscribeError(
            "moonshine-voice is not installed; install audio support: pipx inject scriba moonshine-voice"
        ) from exc

    if isinstance(raw_value, ModelArch):
        return raw_value

    raw_text = str(raw_value).strip()
    if raw_text == "":
        raise TranscribeError("model architecture cannot be empty")

    if raw_text.isdigit():
        try:
            return ModelArch(int(raw_text))
        except ValueError as exc:
            raise TranscribeError(f"unsupported model architecture value: {raw_text}") from exc

    arch_name = normalize_arch_name(raw_text)
    if arch_name not in ARCH_NAMES:
        raise TranscribeError(
            "unsupported model architecture "
            f"{raw_text!r}; expected one of {', '.join(sorted(ARCH_NAMES))} or 0..5"
        )

    try:
        return string_to_model_arch(arch_name)
    except ValueError as exc:
        raise TranscribeError(f"unsupported model architecture: {raw_text}") from exc


def default_tiny_arch():
    try:
        from moonshine_voice import ModelArch
    except ImportError as exc:
        raise TranscribeError(
            "moonshine-voice is not installed; install audio support: pipx inject scriba moonshine-voice"
        ) from exc
    return ModelArch.TINY


def resolve_model(language: str, model_path: str | None, model_arch_value: str | None, verbose: bool):
    """Return ``(model_path, model_arch)`` for transcription."""
    try:
        from moonshine_voice import get_model_for_language, model_arch_to_string
    except ImportError as exc:
        raise TranscribeError(
            "moonshine-voice is not installed; install audio support: pipx inject scriba moonshine-voice"
        ) from exc

    model_arch = parse_model_arch(model_arch_value) if model_arch_value else None

    if model_path:
        custom_model_path = Path(model_path).expanduser()
        if model_arch is None:
            raise TranscribeError(
                "custom --model requires --model-arch, SCRIBA_MODEL_ARCH, or config model_arch"
            )
        if not custom_model_path.exists():
            raise TranscribeError(f"model directory not found: {custom_model_path}")
        if not custom_model_path.is_dir():
            raise TranscribeError(f"model path is not a directory: {custom_model_path}")
        if verbose:
            print(
                f"Using model {custom_model_path} ({model_arch_to_string(model_arch)})",
                flush=True,
            )
        return str(custom_model_path), model_arch

    if model_arch is None:
        model_arch = default_tiny_arch()

    if verbose:
        print(
            f"Using language {language} with model arch {model_arch_to_string(model_arch)}",
            flush=True,
        )
    return get_model_for_language(wanted_language=language, wanted_model_arch=model_arch)
