"""LM Studio OpenAI-compatible follow-up backend."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from interview_prompt_agent.errors import BackendUnavailableError
from interview_prompt_agent.followup.base import FollowupBackend


class LMStudioFollowupBackend(FollowupBackend):
    name = "lmstudio"

    def __init__(self, *, url: str, model: str, timeout_seconds: int = 120) -> None:
        self.url = url
        self.model = model
        self.timeout_seconds = timeout_seconds

    def next_question(self, transcript_so_far: str) -> str:
        prompt = (
            "You are an interview prompt agent. Ask one concise follow-up question "
            "that helps the speaker produce raw material for edited video. "
            "Do not explain. Return only the question.\n\n"
            f"Transcript so far:\n{transcript_so_far[-6000:]}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Ask concise interview follow-up questions."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 1024,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BackendUnavailableError(
                "LM Studio local server is not reachable. Start LM Studio's local server "
                f"and load the Gemma model, then retry. URL: {self.url}"
            ) from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendUnavailableError(f"Unexpected LM Studio response: {data}") from exc
        return _clean_question(content)


def _clean_question(content: object) -> str:
    question = " ".join(str(content).strip().split())
    if not question:
        raise BackendUnavailableError(
            "LM Studio returned an empty follow-up. If this is a reasoning model, "
            "increase the output token budget or disable reasoning in the model runtime."
        )
    if not question.endswith("?"):
        question = f"{question}?"
    return question
