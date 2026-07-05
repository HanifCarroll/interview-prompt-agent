"""Configuration for the CLI and runtime backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DONE_PHRASES = ("next question",)


@dataclass(frozen=True)
class AgentConfig:
    vad: str = "ten"
    stt: str = "whisper_cpp"
    tts: str = "chatterbox_turbo"
    followup: str = "lmstudio"
    sample_rate: int = 16_000
    input_device: str | int | None = None
    done_phrases: tuple[str, ...] = DEFAULT_DONE_PHRASES
    silence_after_done_ms: int = 300
    tail_seconds: float = 8.0
    poll_seconds: float = 2.0
    session_dir: Path = Path("sessions")
    initial_question: str = "What should we talk through first?"
    voice_reference: Path | None = None
    moonshine_language: str = "en"
    moonshine_model: str = "small_streaming"
    moonshine_update_interval: float = 0.25
    kokoro_voice: str = "af_heart"
    piper_model_dir: Path | None = None
    supertonic_model_dir: Path | None = None
    tts_num_threads: int = 4
    tts_speaker_id: int = 0
    tts_speed: float = 1.0
    allow_tts_fallback: bool = False
    lmstudio_url: str = "http://localhost:1234/v1/chat/completions"
    lmstudio_model: str = "qwen3-4b-instruct-2507"
    lmstudio_max_tokens: int = 120
    lmstudio_timeout_seconds: int = 45
    timings: bool = False
    stream_transcripts: bool = False


@dataclass(frozen=True)
class RuntimePaths:
    whisper_cli: str = "whisper-cli"
    whisper_model: Path | None = None
    whisper_control_model: Path | None = None
    sherpa_model_dir: Path | None = None
    sherpa_model_kind: str = "auto"
    sherpa_num_threads: int = 2


@dataclass(frozen=True)
class DoctorResult:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
