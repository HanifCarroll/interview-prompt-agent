import pytest

from interview_prompt_agent.followup.static import StaticFollowupBackend


def test_static_followup_rotates_questions() -> None:
    backend = StaticFollowupBackend(("First?", "Second?"))

    assert backend.next_question("anything") == "First?"
    assert backend.next_question("anything") == "Second?"
    assert backend.next_question("anything") == "First?"


def test_static_followup_rejects_empty_question_list() -> None:
    with pytest.raises(ValueError, match="at least one question"):
        StaticFollowupBackend(())
