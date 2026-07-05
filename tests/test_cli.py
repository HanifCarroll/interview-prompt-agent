from interview_prompt_agent import cli
from interview_prompt_agent.cli import build_parser, main


def test_doctor_json_runs(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out
    assert "whisper-cli" in output
    assert "qwen3-4b-lmstudio" in output


def test_help_includes_reference_commands() -> None:
    output = build_parser().format_help()
    assert "record-reference" in output
    assert "make-reference" in output


def test_run_help_includes_done_phrase() -> None:
    subparsers = next(
        action for action in build_parser()._actions if action.dest == "command"
    ).choices

    assert "--done-phrase" in subparsers["run"].format_help()


def test_run_tts_choices_include_fast_sherpa_backends() -> None:
    subparsers = next(
        action for action in build_parser()._actions if action.dest == "command"
    ).choices
    tts_action = next(action for action in subparsers["run"]._actions if action.dest == "tts")

    assert "piper" in tts_action.choices
    assert "supertonic" in tts_action.choices


def test_keyboard_interrupt_exits_cleanly(monkeypatch, capsys) -> None:
    def raise_interrupt(args) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", raise_interrupt)

    assert cli.main(["run"]) == 130
    output = capsys.readouterr().out
    assert "Interrupted. Exiting cleanly." in output


def test_run_merges_default_and_extra_done_phrases(monkeypatch) -> None:
    captured = {}

    class FakeAgent:
        def __init__(self, config, paths) -> None:
            del paths
            captured["done_phrases"] = config.done_phrases

        def run(self, *, max_turns: int):
            del max_turns
            return "sessions/test"

    monkeypatch.setattr(cli, "InterviewAgent", FakeAgent)

    assert cli.main(["run", "--done-phrase", "next"]) == 0
    assert captured["done_phrases"] == ("next question", "next")
