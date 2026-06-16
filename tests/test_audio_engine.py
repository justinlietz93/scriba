from __future__ import annotations

from array import array
from pathlib import Path
from types import SimpleNamespace

from scriba.audio import load_audio_for_batch
from scriba.engine import transcribe_audio_file


def test_wav_batch_path_uses_moonshine_loader(monkeypatch, tmp_path):
    calls = []
    module = SimpleNamespace(
        load_wav_file=lambda path: calls.append(Path(path)) or ([0.0], 16000)
    )
    monkeypatch.setitem(__import__("sys").modules, "moonshine_voice", module)
    audio = tmp_path / "native.wav"
    audio.write_bytes(b"RIFF0000WAVE")

    audio_data, sample_rate = load_audio_for_batch(audio)

    assert audio_data == [0.0]
    assert sample_rate == 16000
    assert calls == [audio]


def test_non_wav_uses_ffmpeg_conversion(monkeypatch, tmp_path):
    calls = {"run": [], "load": []}

    def fake_run(command, capture_output, text):
        calls["run"].append(command)
        output_path = Path(command[-1])
        output_path.write_bytes(b"RIFF0000WAVE")
        return SimpleNamespace(returncode=0, stderr="")

    module = SimpleNamespace(
        load_wav_file=lambda path: calls["load"].append(Path(path)) or ([0.0], 16000)
    )
    monkeypatch.setitem(__import__("sys").modules, "moonshine_voice", module)
    monkeypatch.setattr("scriba.audio.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("scriba.audio.subprocess.run", fake_run)
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"fake")

    audio_data, sample_rate = load_audio_for_batch(audio)

    assert audio_data == [0.0]
    assert sample_rate == 16000
    assert calls["run"]
    assert "-ar" in calls["run"][0]
    assert calls["load"]


def test_large_file_uses_streaming_path(monkeypatch, tmp_path):
    events = []

    class FakeStream:
        def start(self):
            events.append("start")

        def add_audio(self, chunk, sample_rate):
            events.append(("chunk", list(chunk), sample_rate))

        def stop(self):
            events.append("stop")
            return SimpleNamespace(lines=[SimpleNamespace(text="streamed")])

        def close(self):
            events.append("close")

    class FakeTranscriber:
        def __init__(self, *, model_path, model_arch, options=None):
            events.append(("transcriber", model_path, model_arch))
            events.append(("options", options))

        def create_stream(self, update_interval):
            events.append(("create_stream", update_interval))
            return FakeStream()

        def close(self):
            events.append("transcriber_close")

    module = SimpleNamespace(Transcriber=FakeTranscriber)
    monkeypatch.setitem(__import__("sys").modules, "moonshine_voice", module)
    monkeypatch.setattr(
        "scriba.engine.iter_ffmpeg_audio_chunks",
        lambda path, verbose=False: iter([(array("f", [0.1, 0.2]), 16000)]),
    )
    audio = tmp_path / "large.wav"
    audio.write_bytes(b"x" * 64)

    transcript = transcribe_audio_file(
        audio,
        "/fake/model",
        "tiny",
        streaming_threshold_bytes=1,
    )

    assert transcript.lines[0].text == "streamed"
    assert (
        "options",
        {
            "identify_speakers": "false",
            "return_audio_data": "false",
            "word_timestamps": "false",
        },
    ) in events
    assert ("chunk", [0.10000000149011612, 0.20000000298023224], 16000) in events


def test_non_wav_uses_streaming_even_when_small(monkeypatch, tmp_path):
    events = []

    class FakeStream:
        def start(self):
            events.append("start")

        def add_audio(self, chunk, sample_rate):
            events.append(("chunk", len(chunk), sample_rate))

        def stop(self):
            events.append("stop")
            return SimpleNamespace(lines=[SimpleNamespace(text="streamed m4a")])

        def close(self):
            events.append("close")

    class FakeTranscriber:
        def __init__(self, *, model_path, model_arch, options=None):
            events.append(("transcriber", model_path, model_arch))
            events.append(("options", options))

        def create_stream(self, update_interval):
            events.append(("create_stream", update_interval))
            return FakeStream()

        def close(self):
            events.append("transcriber_close")

    module = SimpleNamespace(Transcriber=FakeTranscriber)
    monkeypatch.setitem(__import__("sys").modules, "moonshine_voice", module)
    monkeypatch.setattr(
        "scriba.engine.iter_ffmpeg_audio_chunks",
        lambda path, verbose=False: iter([(array("f", [0.1, 0.2]), 16000)]),
    )
    audio = tmp_path / "small.m4a"
    audio.write_bytes(b"compressed")

    transcript = transcribe_audio_file(audio, "/fake/model", "tiny")

    assert transcript.lines[0].text == "streamed m4a"
    assert ("create_stream", 10.0) in events
    assert ("chunk", 2, 16000) in events
