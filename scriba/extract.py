"""The router.

For every page we already know its embedded text length for free. The
decision is one branch:

    enough real text  -> keep it (instant, lossless)
    too little / none -> rasterize and OCR (the only pages that pay)

OCR is the only slow step, and it is a subprocess, so the GIL is not the
bottleneck: a thread pool fans out tesseract processes while the main
thread rasterizes the next page. Futures are kept bounded so a huge
scanned document never holds more than a few PNGs in memory at once,
which is what keeps this usable on a constrained laptop.
"""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from . import ocr
from .pdf import Pdf

# A page with at least this many non-whitespace characters is treated as
# carrying a real text layer. Tuned to skip near-empty pages (page
# numbers, a stray header) that are effectively scanned.
DEFAULT_OCR_THRESHOLD = 12

ProgressFn = Callable[[int, int, str], None]  # (done, total, method)


@dataclass
class PageResult:
    index: int
    text: str
    method: str  # "text" or "ocr"


@dataclass
class ExtractResult:
    path: Path
    pages: list[PageResult]

    @property
    def text_pages(self) -> int:
        return sum(1 for p in self.pages if p.method == "text")

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.method == "ocr")

    def assemble(self, *, page_markers: bool = False) -> str:
        chunks: list[str] = []
        for p in self.pages:
            body = p.text.strip("\n")
            if page_markers:
                chunks.append(f"===== page {p.index + 1} =====\n{body}")
            else:
                chunks.append(body)
        return "\n\n".join(chunks).rstrip() + "\n"


def default_jobs() -> int:
    return os.cpu_count() or 1


def extract(
    path: str | Path,
    *,
    dpi: int = 300,
    lang: str = "eng",
    psm: int = 3,
    oem: int = 3,
    jobs: int | None = None,
    force_ocr: bool = False,
    no_ocr: bool = False,
    ocr_threshold: int = DEFAULT_OCR_THRESHOLD,
    clean: bool = True,
    sort: bool = False,
    progress: ProgressFn | None = None,
) -> ExtractResult:
    """Extract text from every page, OCR'ing only image-only pages.

    force_ocr  -> OCR every page (use when the text layer is garbage).
    no_ocr     -> never OCR; emit whatever text layer exists.
    clean      -> inhibit synthesized spaces and rejoin line-break
                  hyphens at extraction time (right default for prose).
    sort       -> order blocks for simple multi-column reading.
    """
    if force_ocr and no_ocr:
        raise ValueError("force_ocr and no_ocr are mutually exclusive")
    jobs = jobs or default_jobs()

    with Pdf(path) as pdf:
        total = pdf.page_count

        # Phase 1: read every text layer (cheap), decide the route.
        keep: dict[int, str] = {}
        ocr_indices: list[int] = []
        for i in range(total):
            layer = pdf.text_layer(i, clean=clean, sort=sort)
            if no_ocr:
                keep[i] = layer.text
            elif force_ocr:
                ocr_indices.append(i)
            elif layer.chars >= ocr_threshold:
                keep[i] = layer.text
            else:
                ocr_indices.append(i)

        done = 0
        if progress:
            for i in keep:
                done += 1
                progress(done, total, "text")

        # Phase 2: OCR the image-only pages with bounded parallelism.
        ocr_text: dict[int, str] = {}
        if ocr_indices:
            ocr.tesseract_path()  # fail fast with a clear message
            max_inflight = max(2, jobs * 2)
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                inflight: dict[Future, int] = {}
                pending = iter(ocr_indices)

                def submit_next() -> bool:
                    idx = next(pending, None)
                    if idx is None:
                        return False
                    png = pdf.render_png(idx, dpi)
                    fut = pool.submit(
                        ocr.ocr_png, png, lang=lang, psm=psm, oem=oem
                    )
                    inflight[fut] = idx
                    return True

                for _ in range(max_inflight):
                    if not submit_next():
                        break

                while inflight:
                    finished, _ = wait(
                        inflight, return_when=FIRST_COMPLETED
                    )
                    for fut in finished:
                        idx = inflight.pop(fut)
                        ocr_text[idx] = fut.result()
                        done += 1
                        if progress:
                            progress(done, total, "ocr")
                        submit_next()

    pages = [
        PageResult(
            index=i,
            text=keep.get(i, ocr_text.get(i, "")),
            method="text" if i in keep else "ocr",
        )
        for i in range(total)
    ]
    return ExtractResult(path=Path(path), pages=pages)
