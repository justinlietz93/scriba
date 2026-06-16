"""Dispatch: route an input file to the backend that turns it into text.

Scriba's one job is "anything -> text". The mechanism is a registry
keyed by file extension. A backend is a small adapter that takes a path
plus the parsed CLI args and returns ``(text, detail)``, where detail is
a short human status for the run summary ("12 text, 3 ocr",
"audio -> 480 chars").

Adding a new input type later (images, docx, html) is a new entry here
plus one adapter function. The CLI, file walking, batch loop, and output
writing never change.
"""

from __future__ import annotations

from pathlib import Path

from .errors import ScribaError

PDF_EXTENSIONS = {".pdf"}

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".m4b", ".aac", ".flac",
    ".ogg", ".opus", ".aif", ".aiff", ".wma", ".webm", ".mp4",
}

SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | AUDIO_EXTENSIONS


def kind_for(path: Path) -> str:
    """Return 'pdf' or 'audio' for a path, or raise on unsupported type."""
    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    raise ScribaError(
        f"unsupported file type {ext or '(none)'!r}: {path.name}. "
        f"supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def to_text(path: Path, args, *, quiet: bool = False) -> tuple[str, str]:
    """Convert one file to text using the backend for its type."""
    kind = kind_for(path)
    if kind == "pdf":
        return _pdf_to_text(path, args, quiet=quiet)
    return _audio_to_text(path, args, quiet=quiet)


# --- PDF backend (ported from pdftext) ---------------------------------

def _pdf_to_text(path: Path, args, *, quiet: bool) -> tuple[str, str]:
    import sys

    from . import ocr
    from .extract import extract
    from .reflow import reflow

    last_len = 0

    def progress(done: int, total: int, method: str) -> None:
        nonlocal last_len
        if quiet:
            return
        line = f"\rscriba: {path.name} {done}/{total} pages ({method}) "
        sys.stderr.write(line.ljust(last_len))
        sys.stderr.flush()
        last_len = len(line)

    try:
        result = extract(
            path,
            dpi=args.dpi,
            lang=args.lang,
            psm=args.psm,
            oem=args.oem,
            jobs=args.jobs,
            force_ocr=args.force_ocr,
            no_ocr=args.no_ocr,
            ocr_threshold=args.ocr_threshold,
            clean=not args.native_spacing,
            sort=args.sort,
            progress=None if quiet else progress,
        )
    except (FileNotFoundError, ValueError, ocr.TesseractError) as exc:
        raise ScribaError(str(exc)) from exc

    if not quiet and last_len:
        sys.stderr.write("\r" + " " * last_len + "\r")
        sys.stderr.flush()

    text = result.assemble(page_markers=args.page_markers)
    if args.reflow or args.unwrap:
        text = reflow(text, hyphens=True, unwrap=args.unwrap)
    detail = f"{result.text_pages} text, {result.ocr_pages} ocr"
    return text, detail


# --- Audio backend (ported from transcribe) ---------------------------

def _audio_to_text(path: Path, args, *, quiet: bool) -> tuple[str, str]:
    from .audio import DEFAULT_STREAMING_THRESHOLD_BYTES, ensure_audio_dependencies
    from .config import load_config, resolve_setting
    from .engine import transcribe_audio_file, transcript_to_text
    from .model import resolve_model
    from .resources import configure_resource_limits

    config = load_config()
    configure_resource_limits(
        resolve_setting(args.threads, "SCRIBA_THREADS", config, "threads", None)
    )
    language = resolve_setting(
        args.language, "SCRIBA_LANGUAGE", config, "language", "en"
    )
    model_path = resolve_setting(
        args.model, "SCRIBA_MODEL_PATH", config, "model_path", config.get("model")
    )
    model_arch_value = resolve_setting(
        args.model_arch, "SCRIBA_MODEL_ARCH", config, "model_arch", None
    )

    # Fail before downloading/loading a model if a needed codec is missing.
    ensure_audio_dependencies(path, DEFAULT_STREAMING_THRESHOLD_BYTES)

    resolved_model_path, model_arch = resolve_model(
        language=language or "en",
        model_path=model_path,
        model_arch_value=model_arch_value,
        verbose=args.verbose,
    )
    transcript = transcribe_audio_file(
        path, resolved_model_path, model_arch, verbose=args.verbose
    )
    text = transcript_to_text(transcript)
    return text, f"audio -> {len(text)} chars"
