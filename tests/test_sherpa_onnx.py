from pathlib import Path

from interview_prompt_agent.stt.sherpa_onnx import _resolve_model_layout


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_resolve_transducer_layout(tmp_path: Path) -> None:
    _touch(tmp_path / "tokens.txt")
    _touch(tmp_path / "encoder-epoch-99-avg-1.onnx")
    _touch(tmp_path / "decoder-epoch-99-avg-1.onnx")
    _touch(tmp_path / "joiner-epoch-99-avg-1.onnx")

    layout = _resolve_model_layout(tmp_path, "auto")

    assert layout["kind"] == "transducer"
    assert layout["tokens"] == tmp_path / "tokens.txt"
    assert layout["encoder"] == tmp_path / "encoder-epoch-99-avg-1.onnx"


def test_resolve_whisper_layout(tmp_path: Path) -> None:
    _touch(tmp_path / "base.en-tokens.txt")
    _touch(tmp_path / "base.en-encoder.int8.onnx")
    _touch(tmp_path / "base.en-decoder.int8.onnx")

    layout = _resolve_model_layout(tmp_path, "auto")

    assert layout["kind"] == "whisper"
    assert layout["tokens"] == tmp_path / "base.en-tokens.txt"


def test_resolve_paraformer_layout(tmp_path: Path) -> None:
    _touch(tmp_path / "tokens.txt")
    _touch(tmp_path / "model.int8.onnx")

    layout = _resolve_model_layout(tmp_path, "paraformer")

    assert layout["kind"] == "paraformer"
    assert layout["model"] == tmp_path / "model.int8.onnx"
