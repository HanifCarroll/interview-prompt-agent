"""VAD backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from interview_prompt_agent.models import SpeechSegment


class VADBackend(ABC):
    name: str

    @abstractmethod
    def speech_segments(self, path: Path) -> list[SpeechSegment]:
        """Return speech segments for a mono 16 kHz WAV file."""
