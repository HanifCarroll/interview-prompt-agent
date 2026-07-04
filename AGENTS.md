# Repository Instructions

This is a public-facing Python CLI tool. Keep changes small, typed, and easy to
verify from the command line.

- Use `uv run --extra dev pytest` for tests.
- Use `uv run --extra dev ruff check .` for linting.
- Keep default dependencies light. Heavy local voice integrations belong behind
  optional extras.
- Do not commit session audio, generated transcripts, local model paths, or
  credentials.
