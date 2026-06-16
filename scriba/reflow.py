"""Optional text-only cleanup applied after extraction.

These operate on the assembled string, not the PDF, so they are a
fallback for the cases the PyMuPDF flags miss. They are opt-in because
each carries an irreducible ambiguity that text alone cannot resolve:

* join_hyphens: an end-of-line hyphen is usually a wrap artifact
  ("inher-\\nently" -> "inherently") but is sometimes a real compound
  that happened to land at the margin ("single-\\nended"). The PDF's
  soft-hyphen marker, which would disambiguate, is normally flattened to
  a plain hyphen during extraction, so this cannot be made perfect. Wrap
  hyphens dominate by a large margin, so the join is right far more often
  than not, but it is a tradeoff, not a guarantee.

* unwrap_paragraphs: joins lines within a paragraph into one line,
  treating a blank line as the paragraph break. Good for feeding prose to
  tools that expect unwrapped text; wrong for poetry, code, or tables.
"""

from __future__ import annotations

import re

_EOL_HYPHEN = re.compile(r"([A-Za-z])-\n([a-z])")
# A line that ends a paragraph: blank line follows, or the line looks
# like a heading/short fragment. We keep it simple: only fold a newline
# into a space when both sides are mid-sentence lowercase/word chars.
_WRAP = re.compile(r"([a-z,;])\n([a-z])")


def join_hyphens(text: str) -> str:
    """Rejoin words split by a line-break hyphen. See module note."""
    return _EOL_HYPHEN.sub(r"\1\2", text)


def unwrap_paragraphs(text: str) -> str:
    """Fold soft line wraps inside a paragraph into single spaces.

    Blank lines (paragraph breaks) are preserved. Only joins when the
    break sits between lowercase/punctuation and lowercase, which avoids
    gluing a heading to the next sentence.
    """
    return _WRAP.sub(r"\1 \2", text)


def reflow(text: str, *, hyphens: bool = True, unwrap: bool = False) -> str:
    if hyphens:
        text = join_hyphens(text)
    if unwrap:
        text = unwrap_paragraphs(text)
    return text
