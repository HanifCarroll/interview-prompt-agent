import pytest

from interview_prompt_agent.errors import BackendUnavailableError
from interview_prompt_agent.followup.lmstudio import _clean_question


def test_clean_question_normalizes_spacing() -> None:
    assert _clean_question("  What should happen next? \n") == "What should happen next?"


def test_clean_question_adds_question_mark() -> None:
    assert _clean_question("What should happen next") == "What should happen next?"


def test_clean_question_rejects_empty_content() -> None:
    with pytest.raises(BackendUnavailableError):
        _clean_question("")
