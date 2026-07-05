"""Kokoro ONNX TTS backend."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from interview_prompt_agent.audio.wav import write_pcm16_wav
from interview_prompt_agent.errors import DependencyMissingError
from interview_prompt_agent.tts.base import TTSBackend

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/kokoro-v1.0.int8.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
    "model-files-v1.0/voices-v1.0.bin"
)


class KokoroBackend(TTSBackend):
    name = "kokoro"

    def __init__(
        self,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        model_path: Path | None = None,
        voices_path: Path | None = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.model_path = model_path
        self.voices_path = voices_path
        self._model = None

    def speak(self, text: str) -> None:
        cached_path = self._cache_path(text)
        if cached_path.exists():
            subprocess.run(["afplay", str(cached_path)], check=True)
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = Path(tmp.name)
        self.synthesize(text, path)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, cached_path)
        subprocess.run(["afplay", str(path)], check=True)

    def synthesize(self, text: str, path: Path) -> Path:
        try:
            import numpy as np
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise DependencyMissingError(
                "Kokoro TTS is not installed. Install with: uv pip install kokoro-onnx"
            ) from exc

        model_path, voices_path = self._ensure_model_files()
        if self._model is None:
            print("Loading Kokoro TTS model...", flush=True)
            self._model = Kokoro(str(model_path), str(voices_path))
            print("Kokoro TTS model loaded.", flush=True)

        print("Synthesizing speech with Kokoro...", flush=True)
        samples, sample_rate = self._model.create(text, voice=self.voice, speed=self.speed)
        return _write_float_audio(path, samples, sample_rate, np=np)

    def _ensure_model_files(self) -> tuple[Path, Path]:
        model_path = self.model_path or _default_model_path()
        voices_path = self.voices_path or _default_voices_path()
        _download_if_missing(model_path, KOKORO_MODEL_URL)
        _download_if_missing(voices_path, KOKORO_VOICES_URL)
        return model_path, voices_path

    def _cache_path(self, text: str) -> Path:
        model_path = self.model_path or _default_model_path()
        voices_path = self.voices_path or _default_voices_path()
        cache_key = "|".join(
            [
                "kokoro_v1",
                text,
                self.voice,
                str(self.speed),
                str(model_path.resolve()),
                str(voices_path.resolve()),
            ]
        )
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return Path(".cache/interview-prompt-agent/kokoro/audio") / f"{digest}.wav"


def _default_model_path() -> Path:
    return Path(".cache/interview-prompt-agent/kokoro/kokoro-v1.0.int8.onnx")


def _default_voices_path() -> Path:
    return Path(".cache/interview-prompt-agent/kokoro/voices-v1.0.bin")


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}...", flush=True)
    with (
        urllib.request.urlopen(url, timeout=120) as response,
        tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    print(f"Wrote {path}", flush=True)


def _write_float_audio(path: Path, samples: object, sample_rate: int, *, np) -> Path:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16).tobytes()
    return write_pcm16_wav(path, pcm16, sample_rate=sample_rate, channels=1)
