"""Text-to-speech backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TTSBackend(ABC):
    name: str

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak text to the default audio output."""

    def synthesize(self, text: str, path: Path) -> Path:
        raise NotImplementedError
