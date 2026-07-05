"""Backend factories."""

from __future__ import annotations

from interview_prompt_agent.config import AgentConfig, RuntimePaths
from interview_prompt_agent.followup.base import FollowupBackend
from interview_prompt_agent.followup.lmstudio import LMStudioFollowupBackend
from interview_prompt_agent.stt.base import STTBackend
from interview_prompt_agent.stt.moonshine_streaming import MoonshineStreamingBackend
from interview_prompt_agent.stt.sherpa_onnx import SherpaOnnxBackend
from interview_prompt_agent.stt.whisper_cpp import WhisperCppBackend
from interview_prompt_agent.tts.base import TTSBackend
from interview_prompt_agent.tts.chatterbox_turbo import ChatterboxTurboBackend
from interview_prompt_agent.tts.kokoro import KokoroBackend
from interview_prompt_agent.tts.macos_say import MacOSSayBackend
from interview_prompt_agent.tts.sherpa import PiperBackend, SupertonicBackend
from interview_prompt_agent.vad.base import VADBackend
from interview_prompt_agent.vad.ten import TenVADBackend


def make_vad(name: str) -> VADBackend:
    if name == "ten":
        return TenVADBackend()
    raise ValueError(f"Unknown VAD backend: {name}")


def make_stt(name: str, paths: RuntimePaths, *, control: bool = False) -> STTBackend:
    if name == "whisper_cpp":
        model = paths.whisper_control_model if control else paths.whisper_model
        return WhisperCppBackend(
            whisper_cli=paths.whisper_cli,
            model=model or paths.whisper_model,
        )
    if name == "sherpa_onnx":
        return SherpaOnnxBackend(
            model_dir=paths.sherpa_model_dir,
            model_kind=paths.sherpa_model_kind,
            num_threads=paths.sherpa_num_threads,
        )
    raise ValueError(f"Unknown STT backend: {name}")


def make_control_stt(name: str, paths: RuntimePaths) -> STTBackend:
    return make_stt(name, paths, control=True)


def make_streaming_stt(config: AgentConfig) -> MoonshineStreamingBackend | None:
    if config.stt == "moonshine_streaming":
        return MoonshineStreamingBackend(
            language=config.moonshine_language,
            model=config.moonshine_model,
            update_interval=config.moonshine_update_interval,
            print_transcripts=config.stream_transcripts,
        )
    return None


def make_tts(config: AgentConfig) -> TTSBackend:
    if config.tts == "chatterbox_turbo":
        return ChatterboxTurboBackend(voice_reference=config.voice_reference)
    if config.tts == "kokoro":
        return KokoroBackend(voice=config.kokoro_voice, speed=config.tts_speed)
    if config.tts == "piper":
        return PiperBackend(
            model_dir=config.piper_model_dir,
            num_threads=config.tts_num_threads,
            speed=config.tts_speed,
        )
    if config.tts == "supertonic":
        return SupertonicBackend(
            model_dir=config.supertonic_model_dir,
            num_threads=config.tts_num_threads,
            speaker_id=config.tts_speaker_id,
            speed=config.tts_speed,
        )
    if config.tts == "macos_say":
        return MacOSSayBackend()
    raise ValueError(f"Unknown TTS backend: {config.tts}")


def make_followup(config: AgentConfig) -> FollowupBackend:
    if config.followup == "lmstudio":
        return LMStudioFollowupBackend(
            url=config.lmstudio_url,
            model=config.lmstudio_model,
            max_tokens=config.lmstudio_max_tokens,
            timeout_seconds=config.lmstudio_timeout_seconds,
        )
    raise ValueError(f"Unknown follow-up backend: {config.followup}")
