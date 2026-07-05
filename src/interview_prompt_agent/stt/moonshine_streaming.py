"""Moonshine live streaming ASR integration."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from interview_prompt_agent.audio.live import _resolve_input_device
from interview_prompt_agent.audio.wav import write_pcm16_wav
from interview_prompt_agent.errors import BackendUnavailableError, DependencyMissingError
from interview_prompt_agent.text import phrase_at_end


@dataclass(frozen=True)
class MoonshineTurnResult:
    answer_audio: Path
    control_transcript: str
    final_transcript: str
    done_phrase: str | None
    done_detected_at: float
    audio_saved_at: float


class MoonshineStreamingBackend:
    name = "moonshine_streaming"

    def __init__(
        self,
        *,
        language: str = "en",
        model: str = "small_streaming",
        update_interval: float = 0.25,
        print_transcripts: bool = False,
    ) -> None:
        self.language = language
        self.model = model
        self.update_interval = update_interval
        self.print_transcripts = print_transcripts
        self._transcriber: Any | None = None

    def preload(self) -> None:
        self._get_transcriber()

    def record_until_done(
        self,
        *,
        output_path: Path,
        interrupted_output_path: Path | None = None,
        done_phrases: tuple[str, ...],
        sample_rate: int,
        channels: int,
        input_device: str | int | None,
        silence_after_done_ms: int,
    ) -> MoonshineTurnResult:
        try:
            import numpy as np
            import sounddevice as sd
        except ImportError as exc:
            raise DependencyMissingError(
                "Moonshine streaming needs moonshine-voice, sounddevice, and numpy. "
                "Install with: uv sync --extra live"
            ) from exc

        transcriber = self._get_transcriber()
        stream = transcriber.create_stream(update_interval=self.update_interval)
        listener = _MoonshineTurnListener(done_phrases)
        stream.add_listener(listener)

        audio_queue: queue.Queue[tuple[bytes, Any]] = queue.Queue()
        frames: list[bytes] = []
        worker_error: list[BaseException] = []
        worker_running = threading.Event()
        worker_running.set()

        def audio_callback(
            indata: Any,
            frame_count: int,
            time_info: object,
            status: object,
        ) -> None:
            del frame_count, time_info
            if status:
                pass
            audio = indata.copy().astype(np.float32).reshape(-1)
            pcm = (audio * 32767).astype(np.int16).tobytes()
            audio_queue.put((pcm, audio))

        def audio_worker() -> None:
            while worker_running.is_set() or not audio_queue.empty():
                try:
                    pcm, audio = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                frames.append(pcm)
                try:
                    stream.add_audio(audio, sample_rate)
                except BaseException as exc:  # pragma: no cover - defensive thread boundary
                    worker_error.append(exc)
                    listener.set_error(exc)
                    return

        input_stream = None
        worker = threading.Thread(target=audio_worker, name="moonshine-audio-worker")
        started_at = time.perf_counter()
        interrupted = False
        try:
            device = _resolve_input_device(sd, input_device)
            stream.start()
            worker.start()
            input_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                device=device,
                callback=audio_callback,
            )
            input_stream.start()

            last_printed = ""
            while not listener.done_event.wait(0.05):
                if listener.error is not None:
                    raise BackendUnavailableError(f"Moonshine streaming failed: {listener.error}")
                latest = listener.latest_transcript
                if self.print_transcripts and latest and latest != last_printed:
                    print(f"stream transcript: {latest}", flush=True)
                    last_printed = latest

            if listener.error is not None:
                raise BackendUnavailableError(f"Moonshine streaming failed: {listener.error}")
            done_detected_at = time.perf_counter()
            latest = listener.latest_transcript
            if self.print_transcripts and latest and latest != last_printed:
                print(f"stream transcript: {latest}", flush=True)
            time.sleep(silence_after_done_ms / 1000)
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            if input_stream is not None:
                try:
                    input_stream.abort()
                finally:
                    input_stream.close()
            worker_running.clear()
            if worker.is_alive():
                worker.join(timeout=3)
            try:
                final_transcript = stream.stop()
                listener.ingest_transcript(final_transcript)
            finally:
                stream.close()
            if interrupted and interrupted_output_path is not None and frames:
                write_pcm16_wav(
                    interrupted_output_path,
                    b"".join(frames),
                    sample_rate=sample_rate,
                    channels=channels,
                )
                print(
                    f"\nInterrupted. Saved partial answer audio: {interrupted_output_path}",
                    flush=True,
                )

        if worker_error:
            raise BackendUnavailableError(f"Moonshine streaming failed: {worker_error[0]}")

        final_text = listener.latest_transcript
        write_pcm16_wav(output_path, b"".join(frames), sample_rate=sample_rate, channels=channels)
        return MoonshineTurnResult(
            answer_audio=output_path,
            control_transcript=final_text,
            final_transcript=final_text,
            done_phrase=listener.done_phrase,
            done_detected_at=done_detected_at - started_at,
            audio_saved_at=time.perf_counter() - started_at,
        )

    def _get_transcriber(self) -> Any:
        if self._transcriber is not None:
            return self._transcriber
        try:
            from moonshine_voice import Transcriber, get_model_for_language
        except ImportError as exc:
            raise DependencyMissingError(
                "Moonshine streaming is not installed. Install with: uv sync --extra live"
            ) from exc

        model_arch = _resolve_model_arch(self.model)
        print(
            f"Loading Moonshine {self.model} ASR model for language {self.language}...",
            flush=True,
        )
        model_path, model_arch = get_model_for_language(self.language, model_arch)
        self._transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=self.update_interval,
        )
        print("Moonshine ASR model loaded.", flush=True)
        return self._transcriber


class _MoonshineTurnListener:
    def __init__(self, done_phrases: tuple[str, ...]) -> None:
        self.done_phrases = done_phrases
        self.done_event = threading.Event()
        self.done_phrase: str | None = None
        self.error: BaseException | None = None
        self._lock = threading.Lock()
        self._line_order: list[int] = []
        self._lines: dict[int, str] = {}
        self._latest_transcript = ""

    @property
    def latest_transcript(self) -> str:
        with self._lock:
            return self._latest_transcript

    def set_error(self, error: BaseException) -> None:
        with self._lock:
            self.error = error
        self.done_event.set()

    def ingest_transcript(self, transcript: Any) -> None:
        for line in getattr(transcript, "lines", []) or []:
            self._handle_line(line)

    def on_line_text_changed(self, event: Any) -> None:
        self._handle_line(event.line)

    def on_line_completed(self, event: Any) -> None:
        self._handle_line(event.line)

    def on_error(self, event: Any) -> None:
        self.set_error(event.error)

    def __call__(self, event: Any) -> None:
        event_name = type(event).__name__
        if event_name == "LineTextChanged":
            self.on_line_text_changed(event)
        elif event_name == "LineCompleted":
            self.on_line_completed(event)
        elif event_name == "Error":
            self.on_error(event)

    def _handle_line(self, line: Any) -> None:
        text = " ".join(str(getattr(line, "text", "")).strip().split())
        if not text:
            return
        line_id = int(getattr(line, "line_id", len(self._line_order)))

        with self._lock:
            if line_id not in self._lines:
                self._line_order.append(line_id)
            self._lines[line_id] = text
            self._latest_transcript = " ".join(
                self._lines[id_] for id_ in self._line_order if self._lines.get(id_)
            ).strip()
            done_phrase = (
                phrase_at_end(text, self.done_phrases)
                or phrase_at_end(self._latest_transcript, self.done_phrases)
            )
            if done_phrase is not None and self.done_phrase is None:
                self.done_phrase = done_phrase
                self.done_event.set()


def _resolve_model_arch(model: str) -> Any:
    try:
        from moonshine_voice import ModelArch
    except ImportError as exc:
        raise DependencyMissingError(
            "Moonshine streaming is not installed. Install with: uv sync --extra live"
        ) from exc

    key = model.replace("-", "_").casefold()
    choices = {
        "tiny": ModelArch.TINY,
        "base": ModelArch.BASE,
        "tiny_streaming": ModelArch.TINY_STREAMING,
        "base_streaming": ModelArch.BASE_STREAMING,
        "small_streaming": ModelArch.SMALL_STREAMING,
        "medium_streaming": ModelArch.MEDIUM_STREAMING,
    }
    if key not in choices:
        valid = ", ".join(sorted(choices))
        raise BackendUnavailableError(f"Unknown Moonshine model: {model}. Valid models: {valid}")
    return choices[key]
