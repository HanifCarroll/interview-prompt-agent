import pytest

from interview_prompt_agent.errors import BackendUnavailableError
from interview_prompt_agent.followup.lmstudio import _build_prompt, _clean_question


def test_clean_question_normalizes_spacing() -> None:
    assert _clean_question("  What should happen next? \n") == "What should happen next?"


def test_clean_question_adds_question_mark() -> None:
    assert _clean_question("What should happen next") == "What should happen next?"


def test_clean_question_rejects_empty_content() -> None:
    with pytest.raises(BackendUnavailableError):
        _clean_question("")


def test_build_prompt_uses_prior_questions_to_avoid_repeats() -> None:
    prompt = _build_prompt(
        "Question 1: What should we talk through first?\n"
        "Answer 1: Working out.\n\n"
        "Question 2: What is a typical workout?\n"
        "Answer 2: Bench press and accessories."
    )

    assert "previous questions and answers" in prompt
    assert "avoid repeating yourself" in prompt
    assert "final Answer block is the current answer" in prompt
    assert "Keep the question under 12 words" in prompt
    assert "Question 2: What is a typical workout?" in prompt
