import pytest

from interview_prompt_agent.agent import InterviewAgent, _append_turn_context


class ValidatingBackend:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def validate(self) -> None:
        self.calls += 1
        if self.error is not None:
            raise self.error


def test_validate_runtime_validates_stt_before_session_work() -> None:
    backend = ValidatingBackend()
    agent = InterviewAgent.__new__(InterviewAgent)
    agent.stt = backend
    agent.control_stt = backend

    agent._validate_runtime()

    assert backend.calls == 1


def test_validate_runtime_propagates_validation_errors() -> None:
    agent = InterviewAgent.__new__(InterviewAgent)
    agent.stt = ValidatingBackend(RuntimeError("bad model"))
    agent.control_stt = None

    with pytest.raises(RuntimeError, match="bad model"):
        agent._validate_runtime()


def test_append_turn_context_includes_questions_and_answers() -> None:
    context = _append_turn_context(
        "",
        index=1,
        question="What should we talk through first?",
        answer="Working out.",
    )
    context = _append_turn_context(
        context,
        index=2,
        question="What is a typical workout?",
        answer="Bench press and accessories.",
    )

    assert context == (
        "Question 1: What should we talk through first?\n"
        "Answer 1: Working out.\n\n"
        "Question 2: What is a typical workout?\n"
        "Answer 2: Bench press and accessories."
    )
