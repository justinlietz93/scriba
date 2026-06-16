"""Transcription orchestration."""

from __future__ import annotations

import sys
from pathlib import Path

from .audio import (
    DEFAULT_STREAMING_THRESHOLD_BYTES,
    iter_ffmpeg_audio_chunks,
    load_audio_for_batch,
    should_stream_file,
)
from .errors import TranscribeError
from . import resources


LOW_RESOURCE_TRANSCRIBER_OPTIONS = {
    "identify_speakers": "false",
    "return_audio_data": "false",
    "word_timestamps": "false",
}

PROGRESS_INTERVAL_SECONDS = 30.0


def transcribe_audio_file(
    audio_path: Path,
    model_path: str,
    model_arch,
    *,
    verbose: bool = False,
    streaming_threshold_bytes: int = DEFAULT_STREAMING_THRESHOLD_BYTES,
):
    try:
        from moonshine_voice import Transcriber
    except ImportError as exc:
        raise TranscribeError(
            "moonshine-voice is not installed; install audio support: pipx inject scriba moonshine-voice"
        ) from exc

    use_streaming = should_stream_file(audio_path, streaming_threshold_bytes)
    transcriber = None
    try:
        transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            options=LOW_RESOURCE_TRANSCRIBER_OPTIONS,
        )
        resources.apply_cpu_affinity(resources.configured_threads())
        if use_streaming:
            return transcribe_streaming(transcriber, audio_path, verbose=verbose)

        audio_data, sample_rate = load_audio_for_batch(audio_path, verbose=verbose)
        return transcriber.transcribe_without_streaming(audio_data, sample_rate=sample_rate)
    except TranscribeError:
        raise
    except Exception as exc:
        raise TranscribeError(f"transcription failed: {exc}") from exc
    finally:
        if transcriber is not None and hasattr(transcriber, "close"):
            transcriber.close()


def transcribe_streaming(transcriber, audio_path: Path, *, verbose: bool = False):
    stream = transcriber.create_stream(update_interval=10.0)
    resources.apply_cpu_affinity(resources.configured_threads())
    processed_seconds = 0.0
    next_progress_seconds = PROGRESS_INTERVAL_SECONDS
    print(
        f"Streaming {audio_path.name} through ffmpeg "
        f"(threads={resources.configured_threads()})...",
        file=sys.stderr,
    )
    try:
        stream.start()
        for chunk, sample_rate in iter_ffmpeg_audio_chunks(audio_path, verbose=verbose):
            stream.add_audio(chunk, sample_rate)
            processed_seconds += len(chunk) / sample_rate
            if processed_seconds >= next_progress_seconds:
                minutes, seconds = divmod(int(processed_seconds), 60)
                print(f"Processed {minutes}m{seconds:02d}s of audio...", file=sys.stderr)
                next_progress_seconds += PROGRESS_INTERVAL_SECONDS
        transcript = stream.stop()
        if transcript is None and hasattr(stream, "update_transcription"):
            transcript = stream.update_transcription()
        return transcript
    finally:
        if hasattr(stream, "close"):
            stream.close()


def transcript_to_text(transcript) -> str:
    lines = getattr(transcript, "lines", None) or []
    parts: list[str] = []
    for line in lines:
        text = getattr(line, "text", "")
        if text is None:
            continue
        clean = str(text).strip()
        if clean:
            parts.append(clean)
    return "\n".join(parts)
