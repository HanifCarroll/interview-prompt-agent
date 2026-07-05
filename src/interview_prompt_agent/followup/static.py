"""Static follow-up fallback."""

from __future__ import annotations

from interview_prompt_agent.followup.base import FollowupBackend

DEFAULT_STATIC_QUESTIONS = (
    "What is the clearest example?",
    "What happened next?",
    "What detail should we unpack?",
    "Why does that matter?",
    "What would make this easier to understand?",
)


class StaticFollowupBackend(FollowupBackend):
    name = "static"

    def __init__(self, questions: tuple[str, ...] = DEFAULT_STATIC_QUESTIONS) -> None:
        if not questions:
            raise ValueError("Static follow-up needs at least one question")
        self.questions = questions
        self._index = 0

    def next_question(self, transcript_so_far: str) -> str:
        del transcript_so_far
        question = self.questions[self._index % len(self.questions)]
        self._index += 1
        return question
