"""Live microphone recording helpers."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from interview_prompt_agent.audio.wav import write_pcm16_wav
from interview_prompt_agent.errors import DependencyMissingError

if TYPE_CHECKING:
    import numpy as np


class LiveRecorder:
    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
        device: str | int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._frames: list[bytes] = []
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise DependencyMissingError(
                "Live microphone recording needs sounddevice and numpy. "
                "Install with: uv sync --extra live"
            ) from exc

        def callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            del frames, time_info
            if status:
                pass
            pcm = (indata.copy() * 32767).astype(np.int16).tobytes()
            self._queue.put(pcm)

        def run() -> None:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=self.device,
                callback=callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def drain(self) -> None:
        while True:
            try:
                self._frames.append(self._queue.get_nowait())
            except queue.Empty:
                return

    def snapshot(self, path: Path) -> Path:
        self.drain()
        return write_pcm16_wav(
            path,
            b"".join(self._frames),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )

    def stop(self, path: Path) -> Path:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.drain()
        return write_pcm16_wav(
            path,
            b"".join(self._frames),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )
