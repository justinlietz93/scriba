"""PDF access over PyMuPDF (fitz).

One dependency does both jobs this tool needs:

* read the embedded text layer of a page (`page.get_text`), and
* rasterize a page to PNG bytes (`page.get_pixmap`) when that layer is
  empty and OCR is the only way to recover the text.

Keeping both behind this module means the rest of the package never
touches fitz directly, and swapping the backend later touches one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

# Default extraction flags. fitz.TEXTFLAGS_TEXT (=195) keeps ligatures and
# whitespace and clips to the mediabox. We add:
#   TEXT_INHIBIT_SPACES  - stop synthesizing spaces from glyph gaps. This
#       is the fix for "im pression": loosely tracked or wide glyphs (m, w)
#       push the inter-glyph gap past PyMuPDF's threshold and a false space
#       is inserted mid-word. Inhibiting keeps only real space glyphs, so
#       genuine word spaces survive while the spurious ones vanish.
#   TEXT_DEHYPHENATE     - rejoin a word split by a line-break hyphen, using
#       layout geometry. (Compound hyphens that happen to fall at a line end
#       are ambiguous and may be joined too; see reflow for the text-only
#       counterpart and its limits.)
CLEAN_FLAGS = fitz.TEXTFLAGS_TEXT | fitz.TEXT_INHIBIT_SPACES | fitz.TEXT_DEHYPHENATE
NATIVE_FLAGS = fitz.TEXTFLAGS_TEXT


@dataclass(frozen=True)
class PageText:
    """The embedded text layer of one page.

    `chars` is the stripped length; the router uses it to decide whether
    a page already carries enough real text to skip OCR.
    """

    index: int  # zero-based page number
    text: str

    @property
    def chars(self) -> int:
        return len(self.text.strip())


class Pdf:
    """A thin, context-managed handle to a PDF document.

    Open the document once, read text layers cheaply, and rasterize only
    the pages that actually need OCR.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such file: {self.path}")
        self._doc = fitz.open(self.path)
        if self._doc.is_encrypted:
            # An empty-password authenticate covers the common "owner
            # password only" case where the content is readable anyway.
            if not self._doc.authenticate(""):
                raise ValueError(
                    f"{self.path.name} is password-protected; "
                    "decrypt it first (e.g. qpdf --decrypt)"
                )

    def __enter__(self) -> "Pdf":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._doc.close()

    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def text_layer(
        self,
        index: int,
        *,
        clean: bool = True,
        sort: bool = False,
    ) -> PageText:
        """Return the embedded text of page `index`, no rasterization.

        clean=True inhibits synthesized inter-glyph spaces and rejoins
        line-break hyphens (the right default for prose). clean=False is
        verbatim native extraction. sort=True orders blocks top-to-bottom
        then left-to-right, which helps simple multi-column pages.
        """
        page = self._doc.load_page(index)
        flags = CLEAN_FLAGS if clean else NATIVE_FLAGS
        return PageText(index=index, text=page.get_text("text", flags=flags, sort=sort))

    def render_png(self, index: int, dpi: int) -> bytes:
        """Rasterize page `index` to PNG bytes at `dpi`.

        Grayscale: OCR ignores color, and one channel is a third the
        bytes through the pipe to tesseract. 300 dpi is tesseract's
        documented sweet spot; below ~200 accuracy falls off fast.
        """
        page = self._doc.load_page(index)
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        return pix.tobytes("png")
