"""macOS say fallback TTS."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from interview_prompt_agent.errors import DependencyMissingError
from interview_prompt_agent.tts.base import TTSBackend


class MacOSSayBackend(TTSBackend):
    name = "macos_say"

    def __init__(self, *, voice: str | None = None) -> None:
        self.voice = voice

    def speak(self, text: str) -> None:
        if shutil.which("say") is None:
            raise DependencyMissingError("macOS `say` is not available")
        cmd = ["say"]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(text)
        subprocess.run(cmd, check=True)

    def synthesize(self, text: str, path: Path) -> Path:
        if shutil.which("say") is None:
            raise DependencyMissingError("macOS `say` is not available")
        cmd = ["say", "-o", str(path)]
        if self.voice:
            cmd.extend(["-v", self.voice])
        cmd.append(text)
        subprocess.run(cmd, check=True)
        return path
