import json
from pathlib import Path

from interview_prompt_agent.models import PromptTurn
from interview_prompt_agent.session.writer import SessionWriter


def test_session_writer_records_turn_manifest(tmp_path: Path) -> None:
    writer = SessionWriter(tmp_path)
    writer.add_turn(
        PromptTurn(
            index=1,
            question="What happened?",
            answer_audio=tmp_path / "answer.wav",
            control_transcript="next question",
            final_transcript="answer",
            done_phrase="next question",
        )
    )

    manifest = json.loads((tmp_path / "session.json").read_text())
    assert manifest["turns"][0]["question"] == "What happened?"
    assert manifest["turns"][0]["done_phrase"] == "next question"
