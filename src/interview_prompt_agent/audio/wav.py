"""Small WAV helpers."""

from __future__ import annotations

import wave
from pathlib import Path


def write_pcm16_wav(path: Path, frames: bytes, *, sample_rate: int, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return path


def read_tail(path: Path, *, seconds: float, output_path: Path) -> Path:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        total_frames = wav.getnframes()
        tail_frames = min(total_frames, int(seconds * sample_rate))
        wav.setpos(max(0, total_frames - tail_frames))
        frames = wav.readframes(tail_frames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(frames)
    return output_path
