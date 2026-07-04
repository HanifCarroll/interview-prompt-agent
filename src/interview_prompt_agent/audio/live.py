"""Live microphone recording helpers."""

from __future__ import annotations

import queue
from pathlib import Path
from typing import TYPE_CHECKING, Any

from interview_prompt_agent.audio.wav import write_pcm16_wav
from interview_prompt_agent.errors import BackendUnavailableError, DependencyMissingError

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
        self._stream: Any | None = None

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

        try:
            device = _resolve_input_device(sd, self.device)
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=device,
                callback=callback,
            )
            stream.start()
            self._stream = stream
        except BackendUnavailableError:
            raise
        except Exception as exc:
            if "stream" in locals():
                stream.close()
            self._stream = None
            raise BackendUnavailableError(f"Could not start microphone input: {exc}") from exc

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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.drain()
        return write_pcm16_wav(
            path,
            b"".join(self._frames),
            sample_rate=self.sample_rate,
            channels=self.channels,
        )


def _resolve_input_device(sd: Any, device: str | int | None) -> int | str | None:
    if device is None:
        return None
    if isinstance(device, int):
        sd.query_devices(device, "input")
        return device

    devices = sd.query_devices()
    input_devices: list[tuple[int, str]] = []
    wanted = device.casefold()
    for index, info in enumerate(devices):
        max_input_channels = int(info.get("max_input_channels", 0))
        name = str(info.get("name", ""))
        if max_input_channels <= 0:
            continue
        input_devices.append((index, name))
        if name.casefold() == wanted:
            return index

    matches = [(index, name) for index, name in input_devices if wanted in name.casefold()]
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        options = ", ".join(f"{index}: {name}" for index, name in matches)
        raise BackendUnavailableError(
            f"Input device name is ambiguous: {device}. Matches: {options}"
        )

    options = ", ".join(f"{index}: {name}" for index, name in input_devices)
    raise BackendUnavailableError(f"Input device not found: {device}. Available inputs: {options}")
