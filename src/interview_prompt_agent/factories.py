"""Backend factories."""

from __future__ import annotations

from interview_prompt_agent.config import AgentConfig, RuntimePaths
from interview_prompt_agent.followup.base import FollowupBackend
from interview_prompt_agent.followup.lmstudio import LMStudioFollowupBackend
from interview_prompt_agent.followup.static import StaticFollowupBackend
from interview_prompt_agent.stt.base import STTBackend
from interview_prompt_agent.stt.sherpa_onnx import SherpaOnnxBackend
from interview_prompt_agent.stt.whisper_cpp import WhisperCppBackend
from interview_prompt_agent.tts.base import TTSBackend
from interview_prompt_agent.tts.chatterbox_turbo import ChatterboxTurboBackend
from interview_prompt_agent.tts.macos_say import MacOSSayBackend
from interview_prompt_agent.vad.base import VADBackend
from interview_prompt_agent.vad.ten import TenVADBackend


def make_vad(name: str) -> VADBackend:
    if name == "ten":
        return TenVADBackend()
    raise ValueError(f"Unknown VAD backend: {name}")


def make_stt(name: str, paths: RuntimePaths) -> STTBackend:
    if name == "whisper_cpp":
        return WhisperCppBackend(whisper_cli=paths.whisper_cli, model=paths.whisper_model)
    if name == "sherpa_onnx":
        return SherpaOnnxBackend(
            model_dir=paths.sherpa_model_dir,
            model_kind=paths.sherpa_model_kind,
            num_threads=paths.sherpa_num_threads,
        )
    raise ValueError(f"Unknown STT backend: {name}")


def make_tts(config: AgentConfig) -> TTSBackend:
    if config.tts == "chatterbox_turbo":
        return ChatterboxTurboBackend(voice_reference=config.voice_reference)
    if config.tts == "macos_say":
        return MacOSSayBackend()
    raise ValueError(f"Unknown TTS backend: {config.tts}")


def make_followup(config: AgentConfig) -> FollowupBackend:
    if config.followup == "lmstudio":
        return LMStudioFollowupBackend(url=config.lmstudio_url, model=config.lmstudio_model)
    if config.followup == "static":
        return StaticFollowupBackend()
    raise ValueError(f"Unknown follow-up backend: {config.followup}")
