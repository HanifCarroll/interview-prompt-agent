"""Text normalization and command phrase detection."""

from __future__ import annotations

import re
import string

_WHITESPACE = re.compile(r"\s+")


def normalize_command_text(text: str) -> str:
    lowered = text.casefold()
    lowered = lowered.replace("’", "'")
    table = str.maketrans("", "", string.punctuation.replace("'", ""))
    cleaned = lowered.translate(table)
    cleaned = cleaned.replace("i'm", "im")
    return _WHITESPACE.sub(" ", cleaned).strip()


def phrase_at_end(text: str, phrases: tuple[str, ...]) -> str | None:
    normalized = normalize_command_text(text)
    for phrase in phrases:
        candidate = normalize_command_text(phrase)
        if normalized == candidate or normalized.endswith(f" {candidate}"):
            return phrase
    return None
