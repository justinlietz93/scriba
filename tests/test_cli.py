from __future__ import annotations

import sys
import types
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

from scriba.cli import main
from scriba.dispatch import kind_for
from scriba.errors import ScribaError


# --- dispatch routing --------------------------------------------------

def test_kind_for_pdf():
    assert kind_for(Path("a.pdf")) == "pdf"
    assert kind_for(Path("A.PDF")) == "pdf"


def test_kind_for_audio():
    for ext in ("wav", "mp3", "m4a", "flac", "ogg"):
        assert kind_for(Path(f"clip.{ext}")) == "audio"


def test_kind_for_unsupported():
    with pytest.raises(ScribaError):
        kind_for(Path("notes.docx"))


# --- audio path through the unified CLI (mocked Moonshine) -------------

class FakeModelArch(IntEnum):
    TINY = 0
    BASE = 1
    TINY_STREAMING = 2
    BASE_STREAMING = 3
    SMALL_STREAMING = 4
    MEDIUM_STREAMING = 5


def install_fake_moonshine(monkeypatch, *, text="hello world", calls=None):
    calls = calls if calls is not None else {}
    calls.setdefault("get_model_for_language", [])
    calls.setdefault("transcribers", [])

    module = types.ModuleType("moonshine_voice")
    module.ModelArch = FakeModelArch

    names = {
        "tiny": FakeModelArch.TINY, "base": FakeModelArch.BASE,
        "tiny-streaming": FakeModelArch.TINY_STREAMING,
        "base-streaming": FakeModelArch.BASE_STREAMING,
        "small-streaming": FakeModelArch.SMALL_STREAMING,
        "medium-streaming": FakeModelArch.MEDIUM_STREAMING,
    }
    rev = {v: k for k, v in names.items()}
    module.string_to_model_arch = lambda v: names[v]
    module.model_arch_to_string = lambda v: rev[v]

    def get_model_for_language(*, wanted_language, wanted_model_arch):
        calls["get_model_for_language"].append((wanted_language, wanted_model_arch))
        return "/fake/model", wanted_model_arch

    module.get_model_for_language = get_model_for_language
    module.load_wav_file = lambda path: ([0.0, 0.1], 16000)

    class FakeTranscriber:
        def __init__(self, *, model_path, model_arch, options=None):
            self.options = options or {}
            calls["transcribers"].append(self)

        def transcribe_without_streaming(self, audio_data, sample_rate=16000):
            return SimpleNamespace(lines=[SimpleNamespace(text=text)])

        def close(self):
            pass

    module.Transcriber = FakeTranscriber
    monkeypatch.setitem(sys.modules, "moonshine_voice", module)
    return calls


def write_wav(path: Path):
    path.write_bytes(b"RIFF0000WAVE")


def test_audio_writes_sibling_txt(monkeypatch, tmp_path):
    install_fake_moonshine(monkeypatch, text="hello moon")
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: None)
    audio = tmp_path / "talk.wav"
    write_wav(audio)
    assert main([str(audio), "-q"]) == 0
    assert (tmp_path / "talk.txt").read_text(encoding="utf-8") == "hello moon\n"


def test_audio_out_dir(monkeypatch, tmp_path):
    install_fake_moonshine(monkeypatch, text="x")
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: None)
    write_wav(tmp_path / "a.wav")
    out = tmp_path / "txt"
    assert main([str(tmp_path / "a.wav"), "-D", str(out), "-q"]) == 0
    assert (out / "a.txt").read_text(encoding="utf-8") == "x\n"


def test_env_config_precedence(monkeypatch, tmp_path):
    calls = install_fake_moonshine(monkeypatch)
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: None)
    home = tmp_path / "home"
    cfg = home / ".config" / "scriba"
    cfg.mkdir(parents=True)
    (cfg / "config.toml").write_text(
        "[scriba]\nlanguage = 'ja'\nmodel_arch = 'base'\n", encoding="utf-8"
    )
    write_wav(tmp_path / "s.wav")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SCRIBA_LANGUAGE", "es")
    assert main(["--language", "en", str(tmp_path / "s.wav"), "-q"]) == 0
    assert calls["get_model_for_language"] == [("en", FakeModelArch.BASE)]


def test_numeric_arch(monkeypatch, tmp_path):
    calls = install_fake_moonshine(monkeypatch)
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: None)
    write_wav(tmp_path / "s.wav")
    assert main(["--model-arch", "3", str(tmp_path / "s.wav"), "-q"]) == 0
    assert calls["get_model_for_language"] == [("en", FakeModelArch.BASE_STREAMING)]


def test_non_wav_without_ffmpeg_fails_before_model_load(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "moonshine_voice", raising=False)
    monkeypatch.setattr("scriba.audio.shutil.which", lambda name: None)
    clip = tmp_path / "clip.mp3"
    clip.write_bytes(b"not really mp3")
    assert main([str(clip), "-q"]) == 1


def test_low_resource_options_applied(monkeypatch, tmp_path):
    calls = install_fake_moonshine(monkeypatch)
    affinities = []
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: affinities.append(n))
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    write_wav(tmp_path / "s.wav")
    assert main(["--threads", "2", str(tmp_path / "s.wav"), "-q"]) == 0
    assert calls["transcribers"][0].options == {
        "identify_speakers": "false",
        "return_audio_data": "false",
        "word_timestamps": "false",
    }
    import os
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert affinities and affinities[0] == 2


def test_mixed_batch_routes_both(monkeypatch, tmp_path):
    """A folder with a PDF and a WAV: each routed to its backend."""
    import fitz
    install_fake_moonshine(monkeypatch, text="spoken words")
    monkeypatch.setattr("scriba.resources.apply_cpu_affinity", lambda n: None)
    doc = fitz.open(); pg = doc.new_page()
    pg.insert_text((72, 72), "written words on a page", fontsize=12)
    doc.save(tmp_path / "doc.pdf"); doc.close()
    write_wav(tmp_path / "talk.wav")

    assert main([str(tmp_path), "-q"]) == 0
    assert "written words" in (tmp_path / "doc.txt").read_text()
    assert (tmp_path / "talk.txt").read_text().strip() == "spoken words"
