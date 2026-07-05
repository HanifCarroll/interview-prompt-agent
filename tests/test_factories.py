from pathlib import Path

from interview_prompt_agent.config import RuntimePaths
from interview_prompt_agent.factories import make_control_stt


def test_make_control_stt_prefers_control_whisper_model() -> None:
    backend = make_control_stt(
        "whisper_cpp",
        RuntimePaths(
            whisper_model=Path("ggml-base.en.bin"),
            whisper_control_model=Path("ggml-tiny.en.bin"),
        ),
    )

    assert backend.model == Path("ggml-tiny.en.bin")


def test_make_control_stt_falls_back_to_main_whisper_model() -> None:
    backend = make_control_stt(
        "whisper_cpp",
        RuntimePaths(whisper_model=Path("ggml-base.en.bin")),
    )

    assert backend.model == Path("ggml-base.en.bin")
