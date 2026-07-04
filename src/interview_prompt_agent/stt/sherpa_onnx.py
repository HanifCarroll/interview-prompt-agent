"""sherpa-onnx offline STT backend."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

from interview_prompt_agent.errors import BackendUnavailableError, DependencyMissingError
from interview_prompt_agent.models import Transcript
from interview_prompt_agent.stt.base import STTBackend

if TYPE_CHECKING:
    import numpy as np


class SherpaOnnxBackend(STTBackend):
    name = "sherpa_onnx"

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        model_kind: str = "auto",
        num_threads: int = 2,
    ) -> None:
        self.model_dir = model_dir
        self.model_kind = model_kind
        self.num_threads = num_threads
        self._recognizer: Any | None = None

    def transcribe_file(self, path: Path) -> Transcript:
        recognizer = self._load_recognizer()
        samples, sample_rate = _read_wave_float32(path)
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        recognizer.decode_streams([stream])
        return Transcript(text=" ".join(stream.result.text.strip().split()))

    def _load_recognizer(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        if self.model_dir is None:
            raise BackendUnavailableError("sherpa-onnx needs --sherpa-model-dir PATH")
        if not self.model_dir.exists():
            raise BackendUnavailableError(f"sherpa model directory not found: {self.model_dir}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise DependencyMissingError(
                "sherpa-onnx is not installed. Install with: uv pip install sherpa-onnx"
            ) from exc

        layout = _resolve_model_layout(self.model_dir, self.model_kind)
        common = {
            "tokens": str(layout["tokens"]),
            "num_threads": self.num_threads,
            "decoding_method": "greedy_search",
            "debug": False,
        }
        kind = str(layout["kind"])
        if kind == "transducer":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(layout["encoder"]),
                decoder=str(layout["decoder"]),
                joiner=str(layout["joiner"]),
                sample_rate=16_000,
                feature_dim=80,
                **common,
            )
        elif kind == "whisper":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(layout["encoder"]),
                decoder=str(layout["decoder"]),
                task="transcribe",
                tail_paddings=-1,
                **common,
            )
        elif kind == "paraformer":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(layout["model"]),
                sample_rate=16_000,
                feature_dim=80,
                **common,
            )
        elif kind == "nemo_ctc":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(layout["model"]),
                sample_rate=16_000,
                feature_dim=80,
                **common,
            )
        elif kind == "wenet_ctc":
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_wenet_ctc(
                model=str(layout["model"]),
                sample_rate=16_000,
                feature_dim=80,
                **common,
            )
        else:
            raise BackendUnavailableError(f"Unsupported sherpa model kind: {kind}")
        return self._recognizer


def _resolve_model_layout(model_dir: Path, model_kind: str) -> dict[str, Path | str]:
    tokens = _find_one(model_dir, ["tokens.txt", "*tokens.txt"], "tokens file")
    kind = model_kind
    if kind == "auto":
        if _has_all(model_dir, ["*encoder*.onnx", "*decoder*.onnx", "*joiner*.onnx"]):
            kind = "transducer"
        elif _has_all(model_dir, ["*encoder*.onnx", "*decoder*.onnx"]):
            kind = "whisper"
        else:
            lower_name = model_dir.name.lower()
            if "nemo" in lower_name:
                kind = "nemo_ctc"
            elif "wenet" in lower_name:
                kind = "wenet_ctc"
            else:
                kind = "paraformer"

    if kind == "transducer":
        return {
            "kind": kind,
            "tokens": tokens,
            "encoder": _find_one(model_dir, ["*encoder*.onnx"], "encoder ONNX file"),
            "decoder": _find_one(model_dir, ["*decoder*.onnx"], "decoder ONNX file"),
            "joiner": _find_one(model_dir, ["*joiner*.onnx"], "joiner ONNX file"),
        }
    if kind == "whisper":
        return {
            "kind": kind,
            "tokens": tokens,
            "encoder": _find_one(model_dir, ["*encoder*.onnx"], "whisper encoder ONNX file"),
            "decoder": _find_one(model_dir, ["*decoder*.onnx"], "whisper decoder ONNX file"),
        }
    if kind in {"paraformer", "nemo_ctc", "wenet_ctc"}:
        return {
            "kind": kind,
            "tokens": tokens,
            "model": _find_one(
                model_dir,
                ["model.int8.onnx", "model.onnx", "*.int8.onnx", "*.onnx"],
                f"{kind} ONNX file",
            ),
        }
    raise BackendUnavailableError(f"Unsupported sherpa model kind: {model_kind}")


def _find_one(model_dir: Path, patterns: list[str], label: str) -> Path:
    for pattern in patterns:
        matches = sorted(path for path in model_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    raise BackendUnavailableError(
        f"Could not find {label} in {model_dir}. Checked: {', '.join(patterns)}"
    )


def _has_all(model_dir: Path, patterns: list[str]) -> bool:
    return all(any(path.is_file() for path in model_dir.glob(pattern)) for pattern in patterns)


def _read_wave_float32(path: Path) -> tuple[np.ndarray, int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise DependencyMissingError(
            "sherpa-onnx audio decoding needs numpy. Install with: uv sync --extra live"
        ) from exc

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV input, got sample width {sample_width}")
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.astype(np.float32) / 32768.0, sample_rate
