from pathlib import Path

import numpy as np

from interview_prompt_agent.tts.kokoro import KokoroBackend, _write_float_audio


def test_kokoro_cache_path_changes_by_voice() -> None:
    first = KokoroBackend(voice="af_heart")._cache_path("Question?")
    second = KokoroBackend(voice="am_adam")._cache_path("Question?")

    assert first != second


def test_write_float_audio_creates_pcm_wav(tmp_path: Path) -> None:
    path = tmp_path / "audio.wav"

    _write_float_audio(path, np.array([0.0, 0.5, -0.5], dtype=np.float32), 24_000, np=np)

    assert path.read_bytes().startswith(b"RIFF")
