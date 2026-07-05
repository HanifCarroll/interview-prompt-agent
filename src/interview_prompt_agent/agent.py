"""Live interview agent orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from interview_prompt_agent.audio.live import LiveRecorder
from interview_prompt_agent.audio.wav import read_tail
from interview_prompt_agent.config import AgentConfig, RuntimePaths
from interview_prompt_agent.factories import (
    make_control_stt,
    make_followup,
    make_streaming_stt,
    make_stt,
    make_tts,
    make_vad,
)
from interview_prompt_agent.models import PromptTurn
from interview_prompt_agent.session.writer import SessionWriter
from interview_prompt_agent.text import phrase_at_end


class InterviewAgent:
    def __init__(self, config: AgentConfig, paths: RuntimePaths) -> None:
        self.config = config
        self.paths = paths
        self.vad = make_vad(config.vad)
        self.streaming_stt = make_streaming_stt(config)
        self.stt = None if self.streaming_stt is not None else make_stt(config.stt, paths)
        self.control_stt = (
            None if self.streaming_stt is not None else make_control_stt(config.stt, paths)
        )
        self.tts = make_tts(config)
        self.followup = make_followup(config)

    def run(self, *, max_turns: int = 3) -> Path:
        writer = SessionWriter.create(self.config.session_dir)
        question = self.config.initial_question
        transcript_so_far = ""
        preload = getattr(self.tts, "preload", None)
        if callable(preload):
            preload()
        if self.streaming_stt is not None:
            self.streaming_stt.preload()
        previous_done_detected_at: float | None = None

        for index in range(1, max_turns + 1):
            if previous_done_detected_at is not None and self.config.timings:
                print(
                    f"timing turn {index}: transition_to_question_text="
                    f"{time.perf_counter() - previous_done_detected_at:.2f}s",
                    flush=True,
                )
            print(f"\nQuestion {index}: {question}", flush=True)
            question_started_at = time.perf_counter()
            self.tts.speak(question)
            if self.config.timings:
                print(
                    f"timing turn {index}: tts_and_playback="
                    f"{time.perf_counter() - question_started_at:.2f}s",
                    flush=True,
                )
            answer_path = writer.path(f"answer-{index:03d}.wav")
            if self.streaming_stt is not None:
                print("Recording with Moonshine streaming ASR.", flush=True)
                print("Say 'next question' when this answer is done.", flush=True)
                recording_started_at = time.perf_counter()
                result = self.streaming_stt.record_until_done(
                    output_path=answer_path,
                    done_phrases=self.config.done_phrases,
                    sample_rate=self.config.sample_rate,
                    channels=1,
                    input_device=self.config.input_device,
                    silence_after_done_ms=self.config.silence_after_done_ms,
                )
                done_detected_at = recording_started_at + result.done_detected_at
                done_phrase = result.done_phrase
                control_transcript = result.control_transcript
                final_text = result.final_transcript
                print(f"Detected done phrase: {done_phrase}. Stopping recording...", flush=True)
                print(f"Saved answer audio: {answer_path}", flush=True)
                if self.config.timings:
                    print(
                        f"timing turn {index}: done_detected="
                        f"{result.done_detected_at:.2f}s record_saved="
                        f"{result.audio_saved_at:.2f}s record_total="
                        f"{time.perf_counter() - recording_started_at:.2f}s",
                        flush=True,
                    )
            else:
                control_transcript, final_text, done_phrase = self._record_polling_turn(
                    index=index,
                    answer_path=answer_path,
                    writer=writer,
                )
                done_detected_at = time.perf_counter()
            print(f"final transcript: {final_text}", flush=True)
            transcript_so_far = f"{transcript_so_far}\n\n{final_text}".strip()
            turn = PromptTurn(
                index=index,
                question=question,
                answer_audio=answer_path,
                control_transcript=control_transcript,
                final_transcript=final_text,
                done_phrase=done_phrase,
            )
            writer.add_turn(turn)
            previous_done_detected_at = done_detected_at
            if index == max_turns:
                continue
            print("Asking follow-up model for the next question...", flush=True)
            followup_started_at = time.perf_counter()
            question = self.followup.next_question(transcript_so_far)
            if self.config.timings:
                print(
                    f"timing turn {index}: followup="
                    f"{time.perf_counter() - followup_started_at:.2f}s",
                    flush=True,
                )

        return writer.root

    def _record_polling_turn(
        self,
        *,
        index: int,
        answer_path: Path,
        writer: SessionWriter,
    ) -> tuple[str, str, str | None]:
        recorder = LiveRecorder(
            sample_rate=self.config.sample_rate,
            device=self.config.input_device,
        )
        control_transcript = ""
        done_phrase: str | None = None
        recording_active = False

        try:
            recorder.start()
            recording_active = True
            print("Recording. Say 'next question' when this answer is done.", flush=True)

            while done_phrase is None:
                time.sleep(self.config.poll_seconds)
                snapshot = recorder.snapshot(writer.path(f"answer-{index:03d}.partial.wav"))
                tail = read_tail(
                    snapshot,
                    seconds=self.config.tail_seconds,
                    output_path=writer.path(f"answer-{index:03d}.tail.wav"),
                )
                if not self.vad.speech_segments(tail):
                    continue
                control_transcript = self.control_stt.transcribe_file(tail).text
                print(f"control transcript: {control_transcript}", flush=True)
                done_phrase = phrase_at_end(control_transcript, self.config.done_phrases)

            print(f"Detected done phrase: {done_phrase}. Stopping recording...", flush=True)
            time.sleep(self.config.silence_after_done_ms / 1000)
            recorder.stop(answer_path)
            recording_active = False
        except KeyboardInterrupt:
            if recording_active:
                interrupted_path = writer.path(f"answer-{index:03d}.interrupted.wav")
                try:
                    recorder.stop(interrupted_path)
                    print(
                        f"\nInterrupted. Saved partial answer audio: {interrupted_path}",
                        flush=True,
                    )
                except Exception as exc:  # pragma: no cover - defensive cleanup path
                    print(f"\nInterrupted. Could not save partial audio: {exc}", flush=True)
            raise

        print(f"Saved answer audio: {answer_path}", flush=True)
        print("Transcribing full answer...", flush=True)
        final_text = self.stt.transcribe_file(answer_path).text
        return control_transcript, final_text, done_phrase
