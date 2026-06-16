"""Tesseract wrapper.

No pytesseract, no Pillow. tesseract already reads an image from stdin
and writes text to stdout, so the leanest bridge is a subprocess that
pipes PNG bytes straight in:

    tesseract stdin stdout --psm 3 -l eng

We force OMP_THREAD_LIMIT=1 so a single tesseract process uses one core.
Parallelism is then owned entirely by the caller's worker pool: N
workers => N cores busy, with no thread oversubscription that would
thrash a constrained laptop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache


class TesseractError(RuntimeError):
    """Raised when the tesseract binary is missing or fails."""


@lru_cache(maxsize=1)
def tesseract_path() -> str:
    path = shutil.which("tesseract")
    if path is None:
        raise TesseractError(
            "tesseract binary not found on PATH. "
            "Install it: sudo apt install tesseract-ocr"
        )
    return path


def version() -> str:
    out = subprocess.run(
        [tesseract_path(), "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    first = (out.stdout or out.stderr).splitlines()
    return first[0].strip() if first else "unknown"


def installed_langs() -> list[str]:
    out = subprocess.run(
        [tesseract_path(), "--list-langs"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = out.stdout.splitlines()
    # First line is a header ("List of available languages ...").
    return [ln.strip() for ln in lines[1:] if ln.strip()]


def ocr_png(png: bytes, *, lang: str = "eng", psm: int = 3, oem: int = 3) -> str:
    """OCR a single PNG image, returning recognized text.

    psm 3 = fully automatic page segmentation (the right default for a
    rasterized document page). oem 3 = default engine (LSTM where built).
    """
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    cmd = [
        tesseract_path(),
        "stdin",
        "stdout",
        "--psm",
        str(psm),
        "--oem",
        str(oem),
        "-l",
        lang,
    ]
    proc = subprocess.run(
        cmd,
        input=png,
        capture_output=True,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise TesseractError(f"tesseract failed (code {proc.returncode}): {msg}")
    return proc.stdout.decode("utf-8", "replace")
