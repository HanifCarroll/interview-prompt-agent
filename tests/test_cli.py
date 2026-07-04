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
