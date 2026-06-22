<img width="2172" height="724" alt="scriba_banner" src="https://github.com/user-attachments/assets/e0c25c82-7449-454d-88af-f52d4edc054d" />

# AI Transcription Tool

Turn things into text. One command takes PDFs and audio files and writes
a plain `.txt` for each. PDFs are read from their text layer and OCR'd
only where a page is image-only; audio is transcribed locally with
Moonshine. Built to stay light on constrained hardware.

```bash
scriba report.pdf            # -> report.txt
scriba interview.m4a         # -> interview.txt (transcribed)
scriba ~/inbox/ -R           # batch a whole tree, each file -> its .txt
scriba scan.pdf -o -         # text to stdout
```

## How it works

Scriba is a dispatcher. Each input is routed by file type to a backend:

* **PDF** -> text-layer extraction, with tesseract OCR only on image-only
  pages. A born-digital document costs almost nothing; OCR runs exactly
  where pixels are the only signal.
* **audio** (`.wav .mp3 .m4a .flac .ogg .opus ...`) -> Moonshine Voice
  transcription, with ffmpeg streaming for large or non-WAV files.

The file walking, batch loop, skip-existing logic, and output handling
are shared. Adding a new input type later (images, docx) is one backend
plus one line in the dispatch table; nothing else changes.

## Install

Linux Mint / Ubuntu mark the system Python as externally managed, so
install as a global command with pipx.

PDF support only (light, no ML stack):

```bash
sudo apt install tesseract-ocr        # OCR engine for scanned pages
pipx install ~/global_packages/scriba
```

Add audio transcription when you want it:

```bash
sudo apt install ffmpeg                # for non-WAV / large audio
pipx inject scriba moonshine-voice     # the transcription model
# or install both at once:
pipx install "scriba[audio]"
```

Moonshine downloads its model on first use and caches it; override the
cache with `MOONSHINE_VOICE_CACHE`. Extra OCR languages are apt packages,
e.g. `sudo apt install tesseract-ocr-deu`, then `scriba doc.pdf -l deu`.

## Output

Each input becomes `<stem>.txt` beside it, or under `--out-dir`. A single
input may go to an explicit `-o FILE` or to stdout with `-o -`. The
progress line and summary go to stderr, so `-o -` stays a clean text
stream you can pipe.

## Batch

Pass several files, or a directory. A directory takes its supported files
(`-R` to recurse). One failure does not stop the run; scriba reports a
per-file and final summary and exits non-zero if anything failed.
`--skip-existing` skips inputs whose `.txt` is already present and newer,
so re-running over a growing folder only does the new work.

```bash
scriba ~/papers/ -R --reflow                # all PDFs in a tree
scriba ~/media/ -D ~/text --skip-existing   # mixed PDFs + audio, resume-safe
```

## Options

Common:

| Flag | Meaning | Default |
|------|---------|---------|
| `-o, --out FILE` | output path; `-` for stdout (single input) | `<stem>.txt` |
| `-D, --out-dir DIR` | write all `.txt` into DIR | beside each input |
| `-R, --recurse` | recurse into subdirectories | off |
| `--skip-existing` | skip inputs whose `.txt` is up to date | off |
| `-q, --quiet` | suppress progress | off |

PDF:

| Flag | Meaning | Default |
|------|---------|---------|
| `-d, --dpi N` | rasterization DPI for OCR pages | `300` |
| `-l, --lang L` | tesseract language(s), e.g. `eng+deu` | `eng` |
| `-j, --jobs N` | parallel OCR workers | CPU count |
| `-t, --ocr-threshold N` | min real chars before a text layer is trusted | `12` |
| `--native-spacing` | keep PyMuPDF raw spacing (disable space fix) | off |
| `-s, --sort` | order blocks for simple multi-column pages | off |
| `-r, --reflow` | rejoin line-break hyphens | off |
| `--unwrap` | with `--reflow`, fold soft wraps into paragraphs | off |
| `-f, --force-ocr` / `-n, --no-ocr` | OCR everything / never OCR | off |

Audio:

| Flag | Meaning | Default |
|------|---------|---------|
| `--language L` | transcription language | `en` |
| `--model DIR` | custom Moonshine model directory | bundled |
| `--model-arch A` | `tiny`, `base`, `*-streaming`, or `0..5` | `tiny` |
| `--threads N` | thread cap for ffmpeg and math runtimes | `2` |

## PDF text-layer quality

A born-digital PDF is only as good as its reconstruction. Scriba fixes
the two defects common in book-quality PDFs:

* **False intra-word spaces** ("im pression"). PyMuPDF synthesizes spaces
  from glyph gaps, and wide glyphs like `m` trip the threshold mid-word.
  Scriba inhibits synthesized spaces by default, keeping only real ones.
  Pass `--native-spacing` to revert.
* **Line-break hyphens** ("inher-\\nently"). `--reflow` rejoins them.

Once a PDF has flattened its soft-hyphen marker, no tool can perfectly
tell a wrap hyphen ("inher-ently") from a compound that landed at the
margin ("single-ended"). Wrap hyphens dominate, so `--reflow` is right
far more often than not, but it is a tradeoff and stays opt-in.

## Migrating from `transcribe`

Scriba subsumes the old `transcribe` command. Differences:

* the command is now `scriba` (audio still works the same way);
* output is written **beside the input** by default, not in the current
  directory (use `-D` to collect elsewhere);
* environment variables are `SCRIBA_LANGUAGE`, `SCRIBA_MODEL_ARCH`,
  `SCRIBA_MODEL_PATH`, `SCRIBA_THREADS`;
* config lives at `~/.config/scriba/config.toml` under a `[scriba]` table;
* `--no-overwrite` is replaced by `--skip-existing`.

## Configuration

`~/.config/scriba/config.toml`:

```toml
[scriba]
language = "en"
model_arch = "tiny"
threads = 2
```

Precedence is CLI flag, then environment variable, then config, then
built-in default.

## Test

```bash
pip install pytest pymupdf
PYTHONPATH=. pytest tests/ -q
```

The suite covers dispatch routing, the PDF text/OCR split (with live
OCR), spacing and reflow, and the audio path with a mocked Moonshine so
it runs without the model.

## License

MIT.
