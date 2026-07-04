"""Session artifact writer."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from interview_prompt_agent.models import PromptTurn


class SessionWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.turns: list[PromptTurn] = []

    @classmethod
    def create(cls, base_dir: Path) -> SessionWriter:
        stamp = datetime.now(ZoneInfo("America/Argentina/Buenos_Aires")).strftime(
            "%Y%m%d-%H%M%S"
        )
        return cls(base_dir / stamp)

    def path(self, name: str) -> Path:
        return self.root / name

    def add_turn(self, turn: PromptTurn) -> None:
        self.turns.append(turn)
        self.write_manifest()

    def write_manifest(self) -> Path:
        manifest = {
            "turns": [turn.to_json() for turn in self.turns],
        }
        path = self.root / "session.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return path
