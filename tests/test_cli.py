from interview_prompt_agent import cli
from interview_prompt_agent.cli import build_parser, main


def test_doctor_json_runs(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out
    assert "whisper-cli" in output
    assert "gemma-4-lmstudio" in output


def test_help_includes_reference_commands() -> None:
    output = build_parser().format_help()
    assert "record-reference" in output
    assert "make-reference" in output


def test_keyboard_interrupt_exits_cleanly(monkeypatch, capsys) -> None:
    def raise_interrupt(args) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run", raise_interrupt)

    assert cli.main(["run"]) == 130
    output = capsys.readouterr().out
    assert "Interrupted. Exiting cleanly." in output
