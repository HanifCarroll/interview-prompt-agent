"""Follow-up question generation contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod


class FollowupBackend(ABC):
    name: str

    @abstractmethod
    def next_question(self, transcript_so_far: str) -> str:
        """Return the next interview question."""
