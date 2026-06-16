"""Scriba CLI: turn things into text.

    scriba report.pdf                  # -> report.txt
    scriba talk.m4a                    # -> talk.txt (transcription)
    scriba a.pdf b.wav c.pdf           # batch, each -> its own .txt
    scriba ~/inbox/ -R --skip-existing # whole tree, resume-safe
    scriba scan.pdf -o -               # text to stdout

One command routes each file by type: PDFs go through the text-layer /
OCR pipeline, audio through Moonshine transcription. Output is always a
plain .txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .dispatch import SUPPORTED_EXTENSIONS, to_text
from .errors import ScribaError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scriba",
        description=(
            "Turn files into plain text. PDFs are read from their text "
            "layer and OCR'd only where needed; audio is transcribed with "
            "Moonshine. One .txt out per input."
        ),
    )
    p.add_argument("input", nargs="+", help="files and/or directories")
    p.add_argument(
        "-R", "--recurse", action="store_true",
        help="recurse into subdirectories for directory inputs",
    )
    p.add_argument(
        "-o", "--out", metavar="FILE", default=None,
        help="output path; '-' for stdout (single input only)",
    )
    p.add_argument(
        "-D", "--out-dir", metavar="DIR", default=None,
        help="write all .txt files into DIR instead of beside each input",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="skip an input whose .txt already exists and is newer",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    p.add_argument("--verbose", action="store_true", help="print backend details")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    pdf = p.add_argument_group("PDF options")
    pdf.add_argument("-d", "--dpi", type=int, default=300,
                     help="rasterization DPI for OCR pages (default: 300)")
    pdf.add_argument("-l", "--lang", default="eng",
                     help="tesseract OCR language(s), e.g. eng+deu (default: eng)")
    pdf.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                     help="parallel OCR workers (default: CPU count)")
    pdf.add_argument("--psm", type=int, default=3,
                     help="tesseract page segmentation mode (default: 3)")
    pdf.add_argument("--oem", type=int, default=3,
                     help="tesseract engine mode (default: 3)")
    pdf.add_argument("-t", "--ocr-threshold", type=int, default=12, metavar="N",
                     help="min real chars before a page's text layer is trusted")
    pdf.add_argument("--native-spacing", action="store_true",
                     help="keep PyMuPDF raw spacing (disable the space fix)")
    pdf.add_argument("-s", "--sort", action="store_true",
                     help="order blocks for simple multi-column pages")
    pdf.add_argument("-r", "--reflow", action="store_true",
                     help="rejoin line-break hyphens (compound-word tradeoff)")
    pdf.add_argument("--unwrap", action="store_true",
                     help="with --reflow, fold soft wraps into paragraphs")
    pdf.add_argument("-m", "--page-markers", action="store_true",
                     help="insert '===== page N =====' between pages")
    ocr_mode = pdf.add_mutually_exclusive_group()
    ocr_mode.add_argument("-f", "--force-ocr", action="store_true",
                          help="OCR every page, ignore the text layer")
    ocr_mode.add_argument("-n", "--no-ocr", action="store_true",
                          help="never OCR; emit only the text layer")

    audio = p.add_argument_group("audio options")
    audio.add_argument("--language", default=None, help="transcription language (default: en)")
    audio.add_argument("--model", default=None, help="path to a custom Moonshine model dir")
    audio.add_argument("--model-arch", default=None,
                       help="Moonshine arch: tiny, base, *-streaming, or 0..5")
    audio.add_argument("--threads", default=None,
                       help="thread cap for ffmpeg and math runtimes (default: 2)")
    return p


def collect_inputs(inputs: list[str], recurse: bool) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated list of
    supported files."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recurse else p.glob("*")
            candidates = sorted(
                c for c in it
                if c.is_file() and c.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        else:
            candidates = [p]
        for c in candidates:
            rp = c.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(c)
    return found


def _out_for(path: Path, args) -> Path:
    if args.out_dir:
        return Path(args.out_dir).expanduser() / (path.stem + ".txt")
    return path.with_suffix(".txt")


def _write(text: str, out_path: Path | None) -> None:
    if out_path is None:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    show = not args.quiet

    try:
        files = collect_inputs(args.input, args.recurse)
    except ScribaError as exc:
        print(f"scriba: error: {exc}", file=sys.stderr)
        return exc.exit_code
    if not files:
        print("scriba: error: no supported files found in input", file=sys.stderr)
        return 1

    # Single input: allow -o and stdout.
    if len(files) == 1 and args.out_dir is None:
        path = files[0]
        out_path = None if args.out == "-" else (
            Path(args.out).expanduser() if args.out else path.with_suffix(".txt")
        )
        start = time.perf_counter()
        try:
            text, detail = to_text(path, args, quiet=args.quiet)
        except ScribaError as exc:
            print(f"scriba: error: {exc}", file=sys.stderr)
            return exc.exit_code
        except KeyboardInterrupt:
            print("scriba: interrupted", file=sys.stderr)
            return 130
        _write(text, out_path)
        if show:
            dest = "stdout" if out_path is None else str(out_path)
            print(
                f"scriba: {detail} -> {dest} in {time.perf_counter() - start:.2f}s",
                file=sys.stderr,
            )
        return 0

    # Batch.
    if args.out == "-":
        print("scriba: error: stdout (-o -) needs a single input", file=sys.stderr)
        return 1
    if args.out is not None:
        print("scriba: error: -o sets one file; use --out-dir for batches",
              file=sys.stderr)
        return 1

    total = len(files)
    converted = skipped = failed = 0
    batch_start = time.perf_counter()
    for i, path in enumerate(files, 1):
        out_path = _out_for(path, args)
        if (args.skip_existing and out_path.exists()
                and out_path.stat().st_mtime >= path.stat().st_mtime):
            skipped += 1
            if show:
                print(f"scriba: [{i}/{total}] skip {path.name}", file=sys.stderr)
            continue
        if show:
            print(f"scriba: [{i}/{total}] {path.name}", file=sys.stderr)
        try:
            text, detail = to_text(path, args, quiet=args.quiet)
        except ScribaError as exc:
            failed += 1
            print(f"scriba: [{i}/{total}] FAILED {path.name}: {exc}", file=sys.stderr)
            continue
        except KeyboardInterrupt:
            print("scriba: interrupted", file=sys.stderr)
            return 130
        _write(text, out_path)
        converted += 1
        if show:
            print(f"scriba: [{i}/{total}] {detail} -> {out_path.name}", file=sys.stderr)

    if show:
        print(
            f"scriba: done {converted}/{total} "
            f"({skipped} skipped, {failed} failed) "
            f"in {time.perf_counter() - batch_start:.2f}s",
            file=sys.stderr,
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
