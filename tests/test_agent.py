import pytest

from interview_prompt_agent.agent import InterviewAgent


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
