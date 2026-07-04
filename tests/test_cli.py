from interview_prompt_agent.cli import main


def test_doctor_json_runs(capsys) -> None:
    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out
    assert "whisper-cli" in output
    assert "gemma-4-lmstudio" in output
