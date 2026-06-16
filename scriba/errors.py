"""Shared exception type for Scriba.

The audio subsystem (ported from `transcribe`) raises `TranscribeError`
throughout; keeping the alias avoids churning every call site while the
CLI catches a single `ScribaError`.
"""


class ScribaError(Exception):
    """User-facing error carrying a process exit code."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


# Backwards-compatible name used by the audio modules.
TranscribeError = ScribaError
