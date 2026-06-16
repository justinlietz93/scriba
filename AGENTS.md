# Scriba

**Goal.** One installable Linux command, `scriba`, that turns files into
plain `.txt`. It dispatches each input by type to a backend: PDFs through
a text-layer + selective-OCR pipeline, audio through local Moonshine
transcription. Designed to stay light on constrained hardware.

**Architecture.**
- `cli.py` — argparse, input collection (files/dirs, `-R`), single vs
  batch, shared output writing, `--skip-existing`.
- `dispatch.py` — extension registry; `kind_for(path)` and
  `to_text(path, args)` route to the PDF or audio backend. New input
  types are added here.
- PDF backend: `extract.py` (route per page: trust text layer or OCR),
  `pdf.py` (PyMuPDF text + rasterize, space-inhibit + dehyphenate flags),
  `ocr.py` (tesseract subprocess), `reflow.py` (opt-in hyphen/wrap fixup).
- Audio backend: `engine.py` (batch/stream orchestration), `audio.py`
  (ffmpeg + WAV loading), `model.py` (Moonshine arch resolution),
  `config.py` (TOML + env precedence), `resources.py` (thread + affinity
  caps).
- `errors.py` — `ScribaError` (with `TranscribeError` alias for the audio
  modules).

**Dependencies.** Core: `pymupdf`. System: `tesseract-ocr` (PDF OCR),
`ffmpeg` (non-WAV/large audio). Optional extra `[audio]`:
`moonshine-voice`, lazily imported so a PDF-only install carries no ML
stack.

**Invariants.**
- Output is one `.txt` per input, beside it by default.
- The PDF path never OCRs a page that already has a real text layer.
- Backends raise `ScribaError`; the CLI catches exactly that.
- Adding an input type touches `dispatch.py` and one adapter only.
