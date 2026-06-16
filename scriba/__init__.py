"""Scriba: turn things into text.

One command, many inputs. PDFs are read from their text layer and OCR'd
only where the page is image-only; audio is transcribed locally with
Moonshine. Every input becomes a plain .txt.

    from scriba.dispatch import to_text   # programmatic single-file use
"""

from ._version import __version__

__all__ = ["__version__"]
