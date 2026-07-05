from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from interview_prompt_agent.tts.sherpa import DEFAULT_LEAD_IN_SECONDS, _write_generated_audio


def test_write_generated_audio_adds_lead_in_silence(tmp_path: Path) -> None:
    sample_rate = 10
    path = tmp_path / "prompt.wav"
    audio = SimpleNamespace(
        samples=np.array([0.5, -0.5], dtype=np.float32),
        sample_rate=sample_rate,
    )

    _write_generated_audio(path, audio)

    with wave.open(str(path), "rb") as wav:
        frames = wav.readframes(wav.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    leading_frames = int(sample_rate * DEFAULT_LEAD_IN_SECONDS)

    assert samples[:leading_frames].tolist() == [0] * leading_frames
    assert np.max(np.abs(samples[leading_frames:])) > 0
