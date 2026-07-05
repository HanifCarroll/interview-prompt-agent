"""Chatterbox Turbo TTS backend."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import warnings
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
        cached_path = self._cache_path(text)
        if cached_path and cached_path.exists():
            subprocess.run(["afplay", str(cached_path)], check=True)
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self.synthesize(text, path)
        if cached_path:
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, cached_path)
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
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            print("Loading Chatterbox Turbo model...", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._model = ChatterboxTurboTTS.from_pretrained(device=self.device)
            except TypeError as exc:
                if "NoneType" in str(exc):
                    raise BackendUnavailableError(
                        "Chatterbox Turbo could not initialize Perth watermarking. "
                        "Install the compatibility dependency with: "
                        "uv pip install 'setuptools<81'"
                    ) from exc
                raise
            print("Chatterbox Turbo model loaded.", flush=True)
        print("Synthesizing speech with Chatterbox Turbo...", flush=True)
        wav = self._model.generate(text, audio_prompt_path=str(self.voice_reference))
        ta.save(str(path), wav, self._model.sr)
        return path

    def _cache_path(self, text: str) -> Path | None:
        if self.voice_reference is None or not self.voice_reference.exists():
            return None
        stat = self.voice_reference.stat()
        cache_key = "|".join(
            [
                "chatterbox_turbo_v1",
                text,
                str(self.voice_reference.resolve()),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            ]
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return Path(".cache/interview-prompt-agent/chatterbox") / f"{digest}.wav"
