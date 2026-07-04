"""Static follow-up fallback."""

from __future__ import annotations

from interview_prompt_agent.followup.base import FollowupBackend


class StaticFollowupBackend(FollowupBackend):
    name = "static"

    def __init__(self, question: str = "What is the most useful next detail?") -> None:
        self.question = question

    def next_question(self, transcript_so_far: str) -> str:
        return self.question
