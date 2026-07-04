"""Chatterbox Turbo TTS backend."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from interview_prompt_agent.errors import BackendUnavailableError, DependencyMissingError
from interview_prompt_agent.tts.base import TTSBackend


class ChatterboxTurboBackend(TTSBackend):
    name = "chatterbox_turbo"

    def __init__(self, *, voice_reference: Path | None = None, device: str = "mps") -> None:
        self.voice_reference = voice_reference
        self.device = device
        self._model = None

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self.synthesize(text, path)
        subprocess.run(["afplay", str(path)], check=True)

    def synthesize(self, text: str, path: Path) -> Path:
        if self.voice_reference is None:
            raise BackendUnavailableError(
                "Chatterbox Turbo needs a short voice reference WAV. Record one with "
                "`interview-agent record-reference` or pass --voice-reference PATH."
            )
        if not self.voice_reference.exists():
            raise BackendUnavailableError(f"Voice reference not found: {self.voice_reference}")
        try:
            import torchaudio as ta
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as exc:
            raise DependencyMissingError(
                "Chatterbox Turbo is not installed. Install with: "
                "uv pip install chatterbox-tts"
            ) from exc
        if self._model is None:
            self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
        wav = self._model.generate(text, audio_prompt_path=str(self.voice_reference))
        ta.save(str(path), wav, self._model.sr)
        return path
