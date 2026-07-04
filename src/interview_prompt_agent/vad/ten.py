"""TEN VAD backend."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import TYPE_CHECKING

from interview_prompt_agent.errors import DependencyMissingError
from interview_prompt_agent.models import SpeechSegment
from interview_prompt_agent.vad.base import VADBackend

if TYPE_CHECKING:
    import numpy as np


class TenVADBackend(VADBackend):
    name = "ten"

    def __init__(
        self,
        *,
        hop_size: int = 256,
        threshold: float = 0.5,
        min_speech_ms: int = 80,
        max_gap_ms: int = 240,
    ) -> None:
        self.hop_size = hop_size
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.max_gap_ms = max_gap_ms

    def speech_segments(self, path: Path) -> list[SpeechSegment]:
        try:
            import numpy as np
            from ten_vad import TenVad
        except ImportError as exc:
            raise DependencyMissingError(
                "TEN VAD is not installed. Install with: "
                "uv pip install -U --force-reinstall -v "
                "git+https://github.com/TEN-framework/ten-vad.git"
            ) from exc

        samples, sample_rate = _read_mono_pcm16(path)
        if sample_rate != 16_000:
            raise ValueError(f"TEN VAD requires 16 kHz WAV input, got {sample_rate}")

        vad = TenVad(self.hop_size, self.threshold)
        frames = len(samples) // self.hop_size
        active_ranges: list[tuple[int, int, float]] = []
        current_start: int | None = None
        current_end: int | None = None
        peak = 0.0

        for frame_index in range(frames):
            start_sample = frame_index * self.hop_size
            frame = samples[start_sample : start_sample + self.hop_size].astype(np.int16)
            probability, flag = vad.process(frame)
            start_ms = int(start_sample * 1000 / sample_rate)
            end_ms = int((start_sample + self.hop_size) * 1000 / sample_rate)
            if flag:
                if current_start is None:
                    current_start = start_ms
                current_end = end_ms
                peak = max(peak, float(probability))
            elif current_start is not None and current_end is not None:
                if start_ms - current_end > self.max_gap_ms:
                    active_ranges.append((current_start, current_end, peak))
                    current_start = None
                    current_end = None
                    peak = 0.0

        if current_start is not None and current_end is not None:
            active_ranges.append((current_start, current_end, peak))

        return [
            SpeechSegment(start_ms=start, end_ms=end, probability=probability)
            for start, end, probability in active_ranges
            if end - start >= self.min_speech_ms
        ]


def _read_mono_pcm16(path: Path) -> tuple[np.ndarray, int]:
    import numpy as np

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
    return samples, sample_rate
