from pathlib import Path

from interview_prompt_agent.tts.chatterbox_turbo import ChatterboxTurboBackend


def test_cache_path_changes_when_reference_changes(tmp_path: Path) -> None:
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"first")
    backend = ChatterboxTurboBackend(voice_reference=reference)
    first = backend._cache_path("What should we talk through first?")

    reference.write_bytes(b"second")
    second = backend._cache_path("What should we talk through first?")

    assert first is not None
    assert second is not None
    assert first != second


def test_cache_path_is_none_without_reference() -> None:
    backend = ChatterboxTurboBackend()

    assert backend._cache_path("Question?") is None
