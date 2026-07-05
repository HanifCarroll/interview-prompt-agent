from __future__ import annotations

import pytest

from interview_prompt_agent.audio.live import _resolve_input_device
from interview_prompt_agent.errors import BackendUnavailableError


class FakeSoundDevice:
    def __init__(self) -> None:
        self.devices = [
            {"name": "Hanif's iPhone Microphone", "max_input_channels": 1},
            {"name": "MacBook Pro Microphone", "max_input_channels": 1},
            {"name": "MacBook Pro Speakers", "max_input_channels": 0},
        ]

    def query_devices(self, device: int | None = None, kind: str | None = None):
        if device is None:
            return self.devices
        if device >= len(self.devices):
            raise RuntimeError("invalid device")
        info = self.devices[device]
        if kind == "input" and info["max_input_channels"] <= 0:
            raise RuntimeError("not an input")
        return info


def test_resolve_input_device_by_name() -> None:
    assert _resolve_input_device(FakeSoundDevice(), "MacBook Pro Microphone") == 1


def test_resolve_input_device_by_unique_partial_name() -> None:
    assert _resolve_input_device(FakeSoundDevice(), "iPhone") == 0


def test_resolve_input_device_rejects_missing_name() -> None:
    with pytest.raises(BackendUnavailableError):
        _resolve_input_device(FakeSoundDevice(), "Studio Display")


def test_live_recorder_stop_aborts_stream_before_close(tmp_path) -> None:
    from interview_prompt_agent.audio.live import LiveRecorder

    class FakeStream:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def abort(self) -> None:
            self.calls.append("abort")

        def close(self) -> None:
            self.calls.append("close")

    stream = FakeStream()
    recorder = LiveRecorder()
    recorder._stream = stream

    recorder.stop(tmp_path / "answer.wav")

    assert stream.calls == ["abort", "close"]
