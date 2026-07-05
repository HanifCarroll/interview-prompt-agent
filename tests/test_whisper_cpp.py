from pathlib import Path

import pytest

from interview_prompt_agent.errors import BackendUnavailableError
from interview_prompt_agent.stt.whisper_cpp import WhisperCppBackend, _clean_output


def test_clean_output_strips_whisper_cpp_noise() -> None:
    output = """
load_backend: loaded CPU backend
ggml_metal_device_init: GPU name: MTL0
read_audio_data: reading audio data

  The actual transcript.
"""
    assert _clean_output(output) == "The actual transcript."


def test_validate_rejects_missing_model(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/local/bin/{name}")

    backend = WhisperCppBackend(model=Path("/path/to/ggml-small.bin"))

    with pytest.raises(BackendUnavailableError, match="Whisper model not found"):
        backend.validate()
