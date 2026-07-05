from pathlib import Path

from interview_prompt_agent.config import AgentConfig, RuntimePaths
from interview_prompt_agent.factories import make_control_stt, make_streaming_stt, make_tts
from interview_prompt_agent.tts.sherpa import PiperBackend, SupertonicBackend


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


def test_make_streaming_stt_returns_moonshine_backend() -> None:
    backend = make_streaming_stt(
        AgentConfig(
            stt="moonshine_streaming",
            moonshine_language="en",
            moonshine_model="tiny_streaming",
            moonshine_update_interval=0.1,
            stream_transcripts=True,
        )
    )

    assert backend is not None
    assert backend.language == "en"
    assert backend.model == "tiny_streaming"
    assert backend.update_interval == 0.1
    assert backend.print_transcripts is True


def test_make_tts_returns_piper_backend() -> None:
    backend = make_tts(
        AgentConfig(
            tts="piper",
            tts_num_threads=6,
            tts_speed=1.2,
            piper_model_dir=Path("models/piper"),
        )
    )

    assert isinstance(backend, PiperBackend)
    assert backend.num_threads == 6
    assert backend.speed == 1.2
    assert backend.model_dir == Path("models/piper")


def test_make_tts_returns_supertonic_backend() -> None:
    backend = make_tts(
        AgentConfig(
            tts="supertonic",
            tts_num_threads=6,
            tts_speaker_id=3,
            tts_speed=1.2,
            supertonic_model_dir=Path("models/supertonic"),
        )
    )

    assert isinstance(backend, SupertonicBackend)
    assert backend.num_threads == 6
    assert backend.speaker_id == 3
    assert backend.speed == 1.2
    assert backend.model_dir == Path("models/supertonic")


def test_make_tts_uses_supertonic_speaker_zero_by_default() -> None:
    backend = make_tts(AgentConfig(tts="supertonic"))

    assert isinstance(backend, SupertonicBackend)
    assert backend.speaker_id == 0
