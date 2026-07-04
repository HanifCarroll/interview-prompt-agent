"""whisper.cpp CLI backend."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from interview_prompt_agent.errors import DependencyMissingError
from interview_prompt_agent.models import Transcript
from interview_prompt_agent.stt.base import STTBackend


class WhisperCppBackend(STTBackend):
    name = "whisper_cpp"

    def __init__(
        self,
        *,
        whisper_cli: str = "whisper-cli",
        model: Path | None = None,
        language: str = "en",
        timeout_seconds: int = 60,
    ) -> None:
        self.whisper_cli = whisper_cli
        self.model = model
        self.language = language
        self.timeout_seconds = timeout_seconds

    def transcribe_file(self, path: Path) -> Transcript:
        binary = shutil.which(self.whisper_cli)
        if binary is None:
            raise DependencyMissingError(f"`{self.whisper_cli}` was not found in PATH")
        cmd = [binary, "-f", str(path), "-l", self.language, "-nt", "--no-prints"]
        if self.model is not None:
            cmd.extend(["-m", str(self.model)])
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
        return Transcript(text=_clean_output(proc.stdout))


def _clean_output(output: str) -> str:
    lines = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("load_backend:", "ggml_", "read_audio_data:")):
            continue
        lines.append(stripped)
    return " ".join(lines).strip()
