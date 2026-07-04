"""Speech-to-text backend contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from interview_prompt_agent.models import Transcript


class STTBackend(ABC):
    name: str

    @abstractmethod
    def transcribe_file(self, path: Path) -> Transcript:
        """Transcribe an audio file."""
