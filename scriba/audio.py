"""Audio loading and ffmpeg conversion helpers."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import os
from array import array
from collections.abc import Iterator
from pathlib import Path

from .errors import TranscribeError


DEFAULT_SAMPLE_RATE = 16000
DEFAULT_STREAMING_THRESHOLD_BYTES = 100 * 1024 * 1024
STREAM_CHUNK_SECONDS = 5
FFMPEG_FORMAT_EXTENSIONS = {
    ".aac",
    ".aif",
    ".aiff",
    ".flac",
    ".m4a",
    ".m4b",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".webm",
    ".wma",
}


def is_wav(path: Path) -> bool:
    return path.suffix.lower() == ".wav"


def should_stream_file(path: Path, threshold_bytes: int = DEFAULT_STREAMING_THRESHOLD_BYTES) -> bool:
    if not is_wav(path):
        return True
    try:
        return path.stat().st_size > threshold_bytes
    except OSError as exc:
        raise TranscribeError(f"cannot inspect audio file {path}: {exc}") from exc


def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def ffmpeg_threads() -> str:
    return os.environ.get("SCRIBA_THREADS", "2")


def ensure_audio_dependencies(
    path: Path, threshold_bytes: int = DEFAULT_STREAMING_THRESHOLD_BYTES
) -> None:
    if should_stream_file(path, threshold_bytes):
        if ffmpeg_path() is None:
            raise TranscribeError(
                "ffmpeg is required for large-file streaming; install ffmpeg or use a smaller WAV"
            )
        return

    if not is_wav(path):
        if ffmpeg_path() is None:
            raise TranscribeError(
                f"ffmpeg is required to transcribe non-WAV input: {path.suffix or 'unknown format'}"
            )


def load_audio_for_batch(path: Path, verbose: bool = False) -> tuple[list[float], int]:
    """Load audio into memory for normal-sized batch transcription."""
    try:
        from moonshine_voice import load_wav_file
    except ImportError as exc:
        raise TranscribeError(
            "moonshine-voice is not installed; install audio support: pipx inject scriba moonshine-voice"
        ) from exc

    if is_wav(path):
        try:
            if verbose:
                print(f"Loading WAV directly: {path}", file=sys.stderr)
            return load_wav_file(path)
        except Exception as exc:
            if ffmpeg_path() is None:
                raise TranscribeError(f"failed to read WAV file {path}: {exc}") from exc
            if verbose:
                print(f"Direct WAV load failed; normalizing with ffmpeg: {exc}", file=sys.stderr)
            return convert_to_temp_wav_and_load(path, load_wav_file, verbose=verbose)

    return convert_to_temp_wav_and_load(path, load_wav_file, verbose=verbose)


def convert_to_temp_wav_and_load(path: Path, load_wav_file, verbose: bool = False):
    executable = ffmpeg_path()
    if executable is None:
        raise TranscribeError(
            f"ffmpeg is required to transcribe non-WAV input: {path.suffix or 'unknown format'}"
        )

    with tempfile.TemporaryDirectory(prefix="scriba-") as tmpdir:
        temp_wav = Path(tmpdir) / "normalized.wav"
        command = [
            executable,
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-threads",
            ffmpeg_threads(),
            "-i",
            str(path),
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(DEFAULT_SAMPLE_RATE),
            "-f",
            "wav",
            str(temp_wav),
        ]
        if verbose:
            print(f"Normalizing audio with ffmpeg: {' '.join(command)}", file=sys.stderr)
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or "unknown ffmpeg error"
            raise TranscribeError(f"ffmpeg failed to convert {path}: {detail}")
        try:
            return load_wav_file(temp_wav)
        except Exception as exc:
            raise TranscribeError(f"failed to read normalized WAV for {path}: {exc}") from exc


def iter_ffmpeg_audio_chunks(
    path: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    chunk_seconds: int = STREAM_CHUNK_SECONDS,
    verbose: bool = False,
) -> Iterator[tuple[array, int]]:
    executable = ffmpeg_path()
    if executable is None:
        raise TranscribeError("ffmpeg is required for large-file streaming")

    command = [
        executable,
        "-nostdin",
        "-v",
        "error",
        "-threads",
        ffmpeg_threads(),
        "-i",
        str(path),
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    if verbose:
        print(f"Streaming audio with ffmpeg: {' '.join(command)}", file=sys.stderr)

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    assert process.stderr is not None

    chunk_bytes = sample_rate * chunk_seconds * 4
    stderr = b""
    try:
        while True:
            data = process.stdout.read(chunk_bytes)
            if not data:
                break
            usable = len(data) - (len(data) % 4)
            if usable == 0:
                continue
            samples = array("f")
            samples.frombytes(data[:usable])
            if sys.byteorder != "little":
                samples.byteswap()
            yield samples, sample_rate
        stderr = process.stderr.read()
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()

    if return_code != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() or "unknown ffmpeg error"
        raise TranscribeError(f"ffmpeg failed to stream {path}: {detail}")
