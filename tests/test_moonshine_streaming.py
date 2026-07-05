from dataclasses import dataclass

from interview_prompt_agent.stt.moonshine_streaming import _MoonshineTurnListener


@dataclass
class FakeLine:
    line_id: int
    text: str


@dataclass
class FakeEvent:
    line: FakeLine


def test_moonshine_listener_detects_done_phrase_at_end() -> None:
    listener = _MoonshineTurnListener(("next question",))

    listener.on_line_text_changed(FakeEvent(FakeLine(1, "I want to talk about lifting weights")))

    assert listener.done_phrase is None
    assert not listener.done_event.is_set()

    listener.on_line_text_changed(
        FakeEvent(FakeLine(1, "I want to talk about lifting weights. Next question."))
    )

    assert listener.done_phrase == "next question"
    assert listener.done_event.is_set()
    assert listener.latest_transcript == "I want to talk about lifting weights. Next question."


def test_moonshine_listener_preserves_line_order() -> None:
    listener = _MoonshineTurnListener(("next question",))

    listener.on_line_completed(FakeEvent(FakeLine(10, "First thought.")))
    listener.on_line_text_changed(FakeEvent(FakeLine(12, "Second thought.")))
    listener.on_line_text_changed(FakeEvent(FakeLine(12, "Second thought. Next question.")))

    assert listener.latest_transcript == "First thought. Second thought. Next question."
    assert listener.done_phrase == "next question"


def test_moonshine_listener_can_be_registered_as_callable() -> None:
    LineTextChanged = type("LineTextChanged", (), {})
    event = LineTextChanged()
    event.line = FakeLine(1, "Ready for the next question.")
    listener = _MoonshineTurnListener(("next question",))

    listener(event)

    assert listener.done_phrase == "next question"
