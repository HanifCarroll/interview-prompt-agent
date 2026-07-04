"""Shared data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SpeechSegment:
    start_ms: int
    end_ms: int
    probability: float | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass(frozen=True)
class Transcript:
    text: str
    segments: list[SpeechSegment] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptTurn:
    index: int
    question: str
    answer_audio: Path
    control_transcript: str
    final_transcript: str | None
    done_phrase: str | None

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["answer_audio"] = str(self.answer_audio)
        return data
