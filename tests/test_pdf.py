"""Tests for scriba.

Build fixtures on the fly with fitz: one born-digital page (real text
layer) and one image-only page (text rasterized into a picture, no
selectable text). The router must read the first directly and OCR the
second.
"""

from __future__ import annotations

import io

import fitz
import pytest

from scriba.extract import extract
from scriba.cli import main
from pathlib import Path

TEXT = "The void dynamics model is a single bifurcation invariant."


def _make_pdf(tmp_path: Path, *, born_digital_pages: int, image_pages: int) -> Path:
    doc = fitz.open()
    for _ in range(born_digital_pages):
        page = doc.new_page()
        page.insert_text((72, 72), TEXT, fontsize=14)
    for _ in range(image_pages):
        # Render a text page to a pixmap, then paste it as an image into
        # a fresh page so no selectable text survives.
        tmp = fitz.open()
        tp = tmp.new_page()
        tp.insert_text((72, 72), TEXT, fontsize=14)
        pix = tp.get_pixmap(dpi=300)
        png = pix.tobytes("png")
        tmp.close()
        page = doc.new_page()
        page.insert_image(page.rect, stream=png)
    out = tmp_path / "fixture.pdf"
    doc.save(out)
    doc.close()
    return out


def test_born_digital_is_read_not_ocrd(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=2, image_pages=0)
    result = extract(pdf)
    assert result.text_pages == 2
    assert result.ocr_pages == 0
    assert "bifurcation invariant" in result.assemble()


def test_image_page_is_ocrd(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=0, image_pages=1)
    result = extract(pdf)
    assert result.ocr_pages == 1
    # OCR is fuzzy; assert on a distinctive long word it should nail.
    assert "dynamics" in result.assemble().lower()


def test_mixed_routes_each_page_correctly(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=1, image_pages=1)
    result = extract(pdf)
    assert result.text_pages == 1
    assert result.ocr_pages == 1
    methods = [p.method for p in result.pages]
    assert methods == ["text", "ocr"]


def test_no_ocr_skips_image_pages(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=0, image_pages=1)
    result = extract(pdf, no_ocr=True)
    assert result.ocr_pages == 0
    assert result.text_pages == 1  # counted as text, just empty


def test_force_ocr_ocrs_everything(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=2, image_pages=0)
    result = extract(pdf, force_ocr=True)
    assert result.ocr_pages == 2
    assert result.text_pages == 0


def test_force_and_no_ocr_conflict(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    with pytest.raises(ValueError):
        extract(pdf, force_ocr=True, no_ocr=True)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        extract("/no/such/file.pdf")


def test_page_markers(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=2, image_pages=0)
    text = extract(pdf).assemble(page_markers=True)
    assert "===== page 1 =====" in text
    assert "===== page 2 =====" in text



def test_cli_writes_sibling_txt(tmp_path):
    pdf = _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    rc = main([str(pdf), "-q"])
    assert rc == 0
    assert pdf.with_suffix(".txt").read_text().strip()


def test_cli_missing_file_returns_1(capsys):
    rc = main(["/no/such/file.pdf", "-q"])
    assert rc == 1


# --- spacing / reflow (added in 0.2.0) ---

from scriba.reflow import join_hyphens, unwrap_paragraphs


def _glyph_run_pdf(tmp_path):
    """A page whose word is laid down glyph-by-glyph with a wide gap after
    'm' and no space glyph, then a real-spaced second word. Reproduces the
    'im pression' false-space class."""
    doc = fitz.open()
    page = doc.new_page()
    tw = fitz.TextWriter(page.rect)
    font = fitz.Font("helv")
    x, y = 72, 100
    for ch in "impression":
        tw.append((x, y), ch, font=font, fontsize=12)
        x += font.glyph_advance(ord(ch)) * 12 + (2.2 if ch == "m" else 0.0)
    tw.append((x + 6, y), " themselves", font=font, fontsize=12)
    tw.write_text(page)
    out = tmp_path / "glyphrun.pdf"
    doc.save(out)
    doc.close()
    return out


def test_clean_spacing_removes_false_intraword_space(tmp_path):
    pdf = _glyph_run_pdf(tmp_path)
    clean = extract(pdf).assemble()
    assert "impression" in clean
    assert "im pression" not in clean


def test_native_spacing_preserves_the_defect(tmp_path):
    pdf = _glyph_run_pdf(tmp_path)
    native = extract(pdf, clean=False).assemble()
    # native reproduces PyMuPDF's heuristic; the false space is present
    assert "im pression" in native


def test_join_hyphens_wrap_case():
    assert join_hyphens("inher-\nently") == "inherently"


def test_join_hyphens_leaves_inline_compound():
    # no newline -> not a wrap, must be untouched
    assert join_hyphens("single-ended") == "single-ended"


def test_unwrap_paragraphs_folds_soft_wraps():
    src = "this line wraps\nonto the next\n\nnew paragraph"
    out = unwrap_paragraphs(src)
    assert "wraps onto the" in out
    assert "\n\nnew paragraph" in out


# --- batch mode (added in 0.3.0) ---

from scriba.cli import collect_inputs


def test_collect_pdfs_dir_nonrecursive(tmp_path):
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4")
    sub = tmp_path / "sub"; sub.mkdir()
    (sub / "c.pdf").write_bytes(b"%PDF-1.4")
    flat = collect_inputs([str(tmp_path)], recurse=False)
    assert {p.name for p in flat} == {"a.pdf", "b.pdf"}
    deep = collect_inputs([str(tmp_path)], recurse=True)
    assert {p.name for p in deep} == {"a.pdf", "b.pdf", "c.pdf"}


def test_collect_pdfs_dedupes(tmp_path):
    f = tmp_path / "a.pdf"; f.write_bytes(b"%PDF-1.4")
    got = collect_inputs([str(f), str(f), str(tmp_path)], recurse=False)
    assert len(got) == 1


def test_batch_writes_sibling_txt(tmp_path):
    a = _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    b = tmp_path / "second.pdf"
    a.replace(tmp_path / "first.pdf")
    _make_pdf(tmp_path, born_digital_pages=1, image_pages=0).replace(b)
    rc = main([str(tmp_path / "first.pdf"), str(b), "-q"])
    assert rc == 0
    assert (tmp_path / "first.txt").read_text().strip()
    assert (tmp_path / "second.txt").read_text().strip()


def test_batch_out_dir(tmp_path):
    _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    outdir = tmp_path / "txt"
    rc = main([str(tmp_path), "-D", str(outdir), "-q"])
    assert rc == 0
    assert list(outdir.glob("*.txt"))


def test_batch_rejects_stdout(tmp_path):
    a = _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    b = tmp_path / "two.pdf"
    _make_pdf(tmp_path, born_digital_pages=1, image_pages=0).replace(b)
    rc = main([str(a), str(b), "-o", "-", "-q"])
    assert rc == 1


def test_skip_existing(tmp_path):
    import shutil

    pdf = _make_pdf(tmp_path, born_digital_pages=1, image_pages=0)
    assert main([str(pdf), "-q"]) == 0          # creates .txt
    txt = pdf.with_suffix(".txt")
    mtime = txt.stat().st_mtime
    other = tmp_path / "other.pdf"
    shutil.copy(pdf, other)
    assert main([str(pdf), str(other), "--skip-existing", "-q"]) == 0
    assert txt.stat().st_mtime == mtime         # untouched
    assert other.with_suffix(".txt").exists()   # the new one was made
