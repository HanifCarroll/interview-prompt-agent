"""Command line interface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from interview_prompt_agent.agent import InterviewAgent
from interview_prompt_agent.config import DEFAULT_DONE_PHRASES, AgentConfig, RuntimePaths
from interview_prompt_agent.errors import AgentError
from interview_prompt_agent.factories import make_followup, make_stt, make_tts, make_vad


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args)
        if args.command == "devices":
            return devices()
        if args.command == "run":
            return run(args)
        if args.command == "record-reference":
            return record_reference(args)
        if args.command == "make-reference":
            return make_reference(args)
        if args.command == "ask-followup":
            return ask_followup(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting cleanly.")
        return 130
    except AgentError as exc:
        print(f"error: {exc}")
        return 2
    parser.print_help()
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="interview-agent")
    sub = parser.add_subparsers(dest="command")

    doctor_parser = sub.add_parser("doctor", help="Check local voice dependencies")
    doctor_parser.add_argument("--json", action="store_true")

    sub.add_parser("devices", help="List live audio input and output devices")

    run_parser = sub.add_parser("run", help="Run a live microphone interview session")
    run_parser.add_argument("--session-dir", type=Path, default=Path("sessions"))
    run_parser.add_argument("--max-turns", type=int, default=3)
    run_parser.add_argument("--initial-question", default="What should we talk through first?")
    run_parser.add_argument(
        "--done-phrase",
        action="append",
        default=[],
        help="Additional explicit phrase that ends the current answer. Can be repeated.",
    )
    run_parser.add_argument("--voice-reference", type=Path)
    run_parser.add_argument("--input-device")
    run_parser.add_argument(
        "--stt",
        choices=["whisper_cpp", "sherpa_onnx", "moonshine_streaming"],
        default="whisper_cpp",
    )
    run_parser.add_argument(
        "--tts",
        choices=["chatterbox_turbo", "kokoro", "macos_say"],
        default="chatterbox_turbo",
    )
    run_parser.add_argument("--kokoro-voice", default="af_heart")
    run_parser.add_argument("--followup", choices=["lmstudio", "static"], default="lmstudio")
    run_parser.add_argument("--whisper-cli", default="whisper-cli")
    run_parser.add_argument("--whisper-model", type=Path)
    run_parser.add_argument(
        "--whisper-control-model",
        type=Path,
        help="Optional smaller whisper.cpp model for done-phrase checks",
    )
    run_parser.add_argument("--poll-seconds", type=float, default=2.0)
    run_parser.add_argument("--tail-seconds", type=float, default=8.0)
    run_parser.add_argument("--moonshine-language", default="en")
    run_parser.add_argument(
        "--moonshine-model",
        choices=[
            "tiny",
            "base",
            "tiny_streaming",
            "base_streaming",
            "small_streaming",
            "medium_streaming",
        ],
        default="small_streaming",
    )
    run_parser.add_argument("--moonshine-update-interval", type=float, default=0.25)
    run_parser.add_argument("--sherpa-model-dir", type=Path)
    run_parser.add_argument(
        "--sherpa-model-kind",
        choices=["auto", "transducer", "whisper", "paraformer", "nemo_ctc", "wenet_ctc"],
        default="auto",
    )
    run_parser.add_argument("--sherpa-num-threads", type=int, default=2)
    run_parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1/chat/completions")
    run_parser.add_argument("--lmstudio-model", default="gemma-4-26b-a4b-it")
    run_parser.add_argument("--lmstudio-max-tokens", type=int, default=1024)
    run_parser.add_argument("--timings", action="store_true")
    run_parser.add_argument(
        "--stream-transcripts",
        action="store_true",
        help="Print evolving Moonshine transcripts while recording.",
    )

    ref_parser = sub.add_parser("record-reference", help="Record a Chatterbox voice reference")
    ref_parser.add_argument("output", type=Path)
    ref_parser.add_argument("--seconds", type=float, default=10.0)
    ref_parser.add_argument("--input-device")

    make_ref_parser = sub.add_parser(
        "make-reference",
        help="Generate a neutral Chatterbox voice reference with macOS say",
    )
    make_ref_parser.add_argument("output", type=Path)
    make_ref_parser.add_argument("--voice", help="macOS say voice name")
    make_ref_parser.add_argument(
        "--text",
        default=(
            "This is a neutral local voice reference for an interview prompt agent. "
            "The voice should sound clear, calm, and conversational."
        ),
    )

    follow_parser = sub.add_parser("ask-followup", help="Ask LM Studio for one follow-up")
    follow_parser.add_argument("transcript", type=Path)
    follow_parser.add_argument("--lmstudio-url", default="http://localhost:1234/v1/chat/completions")
    follow_parser.add_argument("--lmstudio-model", default="gemma-4-26b-a4b-it")
    follow_parser.add_argument("--lmstudio-max-tokens", type=int, default=1024)

    return parser


def doctor(args: argparse.Namespace) -> int:
    checks: dict[str, str] = {}
    checks["whisper-cli"] = shutil.which("whisper-cli") or "missing"
    checks["ffmpeg"] = shutil.which("ffmpeg") or "missing"
    checks["afplay"] = shutil.which("afplay") or "missing"

    for package, label in (
        ("sounddevice", "sounddevice"),
        ("ten_vad", "ten_vad"),
        ("chatterbox", "chatterbox"),
        ("kokoro_onnx", "kokoro_onnx"),
        ("moonshine_voice", "moonshine_voice"),
        ("sherpa_onnx", "sherpa_onnx"),
    ):
        if importlib.util.find_spec(package):
            checks[label] = "installed"
        else:
            checks[label] = "missing"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import perth

        if callable(getattr(perth, "PerthImplicitWatermarker", None)):
            checks["perth-watermarker"] = "installed"
        else:
            checks["perth-watermarker"] = "missing; install setuptools<81"
    except ImportError:
        checks["perth-watermarker"] = "missing"

    gemma_path = Path.home() / ".lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-GGUF"
    checks["gemma-4-lmstudio"] = str(gemma_path) if gemma_path.exists() else "missing"

    if args.json:
        print(json.dumps(checks, indent=2))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
    return 0


def devices() -> int:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise AgentError(
            "Audio device listing needs sounddevice. Install with: uv sync --extra live"
        ) from exc
    print(sd.query_devices())
    return 0


def run(args: argparse.Namespace) -> int:
    config = AgentConfig(
        stt=args.stt,
        tts=args.tts,
        followup=args.followup,
        session_dir=args.session_dir,
        initial_question=args.initial_question,
        done_phrases=tuple(dict.fromkeys((*DEFAULT_DONE_PHRASES, *args.done_phrase))),
        voice_reference=args.voice_reference,
        moonshine_language=args.moonshine_language,
        moonshine_model=args.moonshine_model,
        moonshine_update_interval=args.moonshine_update_interval,
        kokoro_voice=args.kokoro_voice,
        input_device=_coerce_device(args.input_device),
        poll_seconds=args.poll_seconds,
        tail_seconds=args.tail_seconds,
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
        lmstudio_max_tokens=args.lmstudio_max_tokens,
        timings=args.timings,
        stream_transcripts=args.stream_transcripts,
    )
    paths = RuntimePaths(
        whisper_cli=args.whisper_cli,
        whisper_model=args.whisper_model,
        whisper_control_model=args.whisper_control_model,
        sherpa_model_dir=args.sherpa_model_dir,
        sherpa_model_kind=args.sherpa_model_kind,
        sherpa_num_threads=args.sherpa_num_threads,
    )
    session_path = InterviewAgent(config, paths).run(max_turns=args.max_turns)
    print(f"Session written to {session_path}")
    return 0


def record_reference(args: argparse.Namespace) -> int:
    from interview_prompt_agent.audio.live import LiveRecorder

    recorder = LiveRecorder(device=_coerce_device(args.input_device))
    print(f"Recording {args.seconds:.1f}s voice reference...")
    recorder.start()
    import time

    time.sleep(args.seconds)
    recorder.stop(args.output)
    print(f"Wrote {args.output}")
    return 0


def make_reference(args: argparse.Namespace) -> int:
    say = shutil.which("say")
    ffmpeg = shutil.which("ffmpeg")
    if say is None:
        raise AgentError("macOS say command was not found")
    if ffmpeg is None:
        raise AgentError("ffmpeg is required to convert the generated reference to WAV")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".aiff") as tmp:
        say_command = [say]
        if args.voice:
            say_command.extend(["-v", args.voice])
        say_command.extend(["-o", tmp.name, args.text])
        try:
            subprocess.run(say_command, check=True)
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    tmp.name,
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    str(args.output),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise AgentError(f"Could not generate voice reference: {exc}") from exc
    print(f"Wrote {args.output}")
    return 0


def ask_followup(args: argparse.Namespace) -> int:
    config = AgentConfig(
        followup="lmstudio",
        lmstudio_url=args.lmstudio_url,
        lmstudio_model=args.lmstudio_model,
        lmstudio_max_tokens=args.lmstudio_max_tokens,
    )
    # Keep factory imports exercised by the public CLI surface.
    make_vad("ten")
    make_stt("whisper_cpp", RuntimePaths())
    make_tts(AgentConfig(tts="macos_say"))
    backend = make_followup(config)
    print(backend.next_question(args.transcript.read_text(encoding="utf-8")))
    return 0


def _coerce_device(value: str | None) -> str | int | None:
    if value is None:
        return None
    if value.isdigit():
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
