"""sherpa-onnx offline TTS backends."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from interview_prompt_agent.audio.wav import write_pcm16_wav
from interview_prompt_agent.errors import BackendUnavailableError, DependencyMissingError
from interview_prompt_agent.tts.base import TTSBackend

PIPER_MODEL_NAME = "vits-piper-en_US-lessac-medium"
PIPER_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    f"{PIPER_MODEL_NAME}.tar.bz2"
)

SUPERTONIC_REPO_BASE = (
    "https://huggingface.co/csukuangfj2/"
    "sherpa-onnx-supertonic-tts-int8-2026-03-06/resolve/main"
)
SUPERTONIC_FILES = (
    "duration_predictor.int8.onnx",
    "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx",
    "vocoder.int8.onnx",
    "tts.json",
    "unicode_indexer.bin",
    "voice.bin",
)


class PiperBackend(TTSBackend):
    name = "piper"

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        num_threads: int = 4,
        speed: float = 1.0,
    ) -> None:
        self.model_dir = model_dir
        self.num_threads = num_threads
        self.speed = speed
        self._model = None

    def preload(self) -> None:
        self._get_model()

    def speak(self, text: str) -> None:
        _speak_cached(self, text, "Piper")

    def synthesize(self, text: str, path: Path) -> Path:
        tts = self._get_model()
        print("Synthesizing speech with Piper...", flush=True)
        audio = tts.generate(text, sid=0, speed=self.speed)
        return _write_generated_audio(path, audio)

    def _get_model(self):
        if self._model is not None:
            return self._model
        sherpa_onnx = _load_sherpa()
        model_dir = self._ensure_model_dir()
        print("Loading Piper TTS model...", flush=True)
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=str(model_dir / "en_US-lessac-medium.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            data_dir=str(model_dir / "espeak-ng-data"),
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            vits=vits,
            num_threads=self.num_threads,
            provider="cpu",
        )
        config = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
        self._model = sherpa_onnx.OfflineTts(config)
        print("Piper TTS model loaded.", flush=True)
        return self._model

    def _ensure_model_dir(self) -> Path:
        model_dir = self.model_dir or _default_piper_model_dir()
        if self.model_dir is None and not _piper_model_ready(model_dir):
            archive_path = model_dir.parent / f"{PIPER_MODEL_NAME}.tar.bz2"
            _download_if_missing(archive_path, PIPER_MODEL_URL)
            _extract_archive(archive_path, model_dir.parent)
        _require_files(
            model_dir,
            (
                "en_US-lessac-medium.onnx",
                "tokens.txt",
                "espeak-ng-data/phontab",
            ),
            "Piper",
        )
        return model_dir

    def _cache_path(self, text: str) -> Path:
        model_dir = self.model_dir or _default_piper_model_dir()
        return _cache_path(
            namespace="piper",
            identity=[
                "piper_vits_lessac_medium_v1",
                text,
                str(self.speed),
                str(model_dir.resolve()),
            ],
        )


class SupertonicBackend(TTSBackend):
    name = "supertonic"

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        num_threads: int = 4,
        speaker_id: int = 0,
        speed: float = 1.0,
    ) -> None:
        self.model_dir = model_dir
        self.num_threads = num_threads
        self.speaker_id = speaker_id
        self.speed = speed
        self._model = None

    def preload(self) -> None:
        self._get_model()

    def speak(self, text: str) -> None:
        _speak_cached(self, text, "Supertonic")

    def synthesize(self, text: str, path: Path) -> Path:
        tts = self._get_model()
        print("Synthesizing speech with Supertonic...", flush=True)
        audio = tts.generate(text, sid=self.speaker_id, speed=self.speed)
        return _write_generated_audio(path, audio)

    def _get_model(self):
        if self._model is not None:
            return self._model
        sherpa_onnx = _load_sherpa()
        model_dir = self._ensure_model_dir()
        print("Loading Supertonic TTS model...", flush=True)
        supertonic = sherpa_onnx.OfflineTtsSupertonicModelConfig(
            duration_predictor=str(model_dir / "duration_predictor.int8.onnx"),
            text_encoder=str(model_dir / "text_encoder.int8.onnx"),
            vector_estimator=str(model_dir / "vector_estimator.int8.onnx"),
            vocoder=str(model_dir / "vocoder.int8.onnx"),
            tts_json=str(model_dir / "tts.json"),
            unicode_indexer=str(model_dir / "unicode_indexer.bin"),
            voice_style=str(model_dir / "voice.bin"),
        )
        model = sherpa_onnx.OfflineTtsModelConfig(
            supertonic=supertonic,
            num_threads=self.num_threads,
            provider="cpu",
        )
        config = sherpa_onnx.OfflineTtsConfig(model=model, max_num_sentences=1)
        self._model = sherpa_onnx.OfflineTts(config)
        print("Supertonic TTS model loaded.", flush=True)
        return self._model

    def _ensure_model_dir(self) -> Path:
        model_dir = self.model_dir or _default_supertonic_model_dir()
        if self.model_dir is None:
            for filename in SUPERTONIC_FILES:
                _download_if_missing(model_dir / filename, f"{SUPERTONIC_REPO_BASE}/{filename}")
        _require_files(model_dir, SUPERTONIC_FILES, "Supertonic")
        return model_dir

    def _cache_path(self, text: str) -> Path:
        model_dir = self.model_dir or _default_supertonic_model_dir()
        return _cache_path(
            namespace="supertonic",
            identity=[
                "supertonic_int8_v1",
                text,
                str(self.speaker_id),
                str(self.speed),
                str(model_dir.resolve()),
            ],
        )


def _speak_cached(backend: PiperBackend | SupertonicBackend, text: str, label: str) -> None:
    cached_path = backend._cache_path(text)
    if cached_path.exists():
        print(f"Playing cached {label} audio...", flush=True)
        subprocess.run(["afplay", str(cached_path)], check=True)
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        path = Path(tmp.name)
    started_at = time.perf_counter()
    backend.synthesize(text, path)
    print(
        f"{label} audio ready in {time.perf_counter() - started_at:.2f}s. Playing...",
        flush=True,
    )
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, cached_path)
    subprocess.run(["afplay", str(path)], check=True)


def _write_generated_audio(path: Path, audio: object) -> Path:
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyMissingError(
            "sherpa-onnx TTS needs numpy. Install with: uv sync --extra live"
        ) from exc
    samples = np.asarray(audio.samples, dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(np.int16).tobytes()
    return write_pcm16_wav(path, pcm16, sample_rate=audio.sample_rate, channels=1)


def _load_sherpa():
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise DependencyMissingError(
            "Piper and Supertonic TTS need sherpa-onnx. Install with: uv pip install sherpa-onnx"
        ) from exc
    return sherpa_onnx


def _download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}...", flush=True)
    with (
        urllib.request.urlopen(url, timeout=300) as response,
        tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    print(f"Wrote {path}", flush=True)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    print(f"Extracting {archive_path.name}...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if os.path.commonpath([str(output_root), str(target)]) != str(output_root):
                raise BackendUnavailableError(
                    f"Archive member escapes output directory: {member.name}"
                )
        archive.extractall(output_dir)
    print(f"Extracted {archive_path.name}", flush=True)


def _require_files(model_dir: Path, relative_paths: tuple[str, ...], label: str) -> None:
    missing = [
        str(model_dir / rel_path)
        for rel_path in relative_paths
        if not (model_dir / rel_path).exists()
    ]
    if missing:
        raise BackendUnavailableError(
            f"{label} model directory is incomplete: {model_dir}. Missing: {', '.join(missing)}"
        )


def _piper_model_ready(model_dir: Path) -> bool:
    return (
        (model_dir / "en_US-lessac-medium.onnx").exists()
        and (model_dir / "tokens.txt").exists()
        and (model_dir / "espeak-ng-data" / "phontab").exists()
    )


def _cache_path(*, namespace: str, identity: list[str]) -> Path:
    digest = hashlib.sha256("|".join(identity).encode("utf-8")).hexdigest()
    return Path(".cache/interview-prompt-agent") / namespace / "audio" / f"{digest}.wav"


def _default_piper_model_dir() -> Path:
    return Path(".cache/interview-prompt-agent/sherpa-tts/piper") / PIPER_MODEL_NAME


def _default_supertonic_model_dir() -> Path:
    return Path(".cache/interview-prompt-agent/sherpa-tts/supertonic-int8")
