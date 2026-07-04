from interview_prompt_agent.text import normalize_command_text, phrase_at_end


def test_normalize_command_text_handles_punctuation_and_apostrophes() -> None:
    assert normalize_command_text("Next question.") == "next question"
    assert normalize_command_text("I’m done!") == "im done"


def test_phrase_at_end_requires_final_phrase() -> None:
    assert phrase_at_end("That is the point. Next question.", ("next question",)) == (
        "next question"
    )
    assert (
        phrase_at_end("Next question is an interesting phrase to test", ("next question",))
        is None
    )


def test_phrase_at_end_ignores_non_configured_done_language() -> None:
    assert phrase_at_end("until I was done speaking", ("next question",)) is None
