# interview-prompt-agent

`interview-prompt-agent` is a local-first voice prompt agent for recording
interview-style raw material.

It asks a question, records from the Mac microphone, waits until the speaker
says `next question`, then generates and speaks the next follow-up.

The main design rule is explicit turn completion. Pauses and long silences are
treated as thinking time, not as permission for the agent to interrupt.

## Status

Alpha. The repo is structured for public use, but the live voice stack depends
on local model/runtime setup:

- TEN VAD for speech activity detection.
- Moonshine streaming ASR, `whisper.cpp`, or `sherpa-onnx` for speech-to-text.
- Supertonic, Piper, Kokoro, or Chatterbox Turbo for local text-to-speech.
- LM Studio with a local chat model for follow-up questions.

## How It Works

```text
question -> local TTS -> microphone recording
                                      |
                                      v
                         TEN VAD checks recent speech windows
                                      |
                                      v
                     STT transcribes only recent control audio
                                      |
                                      v
                     "next question" at the end stops the turn
                                      |
                                      v
                 full answer is transcribed and sent to LM Studio
                                      |
                                      v
                         next follow-up question is spoken
```

VAD means Voice Activity Detection. It detects whether speech is present in an
audio frame. It does not decide whether the speaker is done.

## Install

Clone and install the light development environment:

```sh
uv sync --extra dev
```

Install live microphone support:

```sh
uv sync --extra dev --extra live
```

Install TEN VAD:

```sh
uv pip install -U --force-reinstall -v git+https://github.com/TEN-framework/ten-vad.git
```

Install Chatterbox Turbo:

```sh
uv pip install "setuptools<81" chatterbox-tts
```

`setuptools<81` is needed because Chatterbox's Perth watermark dependency still
imports `pkg_resources`.

Install Kokoro TTS:

```sh
uv pip install kokoro-onnx
```

Install sherpa-onnx support:

```sh
uv pip install sherpa-onnx
```

This enables both sherpa STT and the fast local Piper/Supertonic TTS backends.
The default Piper and Supertonic model files download on first use and are
reused from `.cache/interview-prompt-agent/sherpa-tts/`.

Install Moonshine streaming ASR support:

```sh
uv pip install moonshine-voice
```

Install and load a faster LM Studio follow-up model:

```sh
lms get qwen/qwen3-4b-2507 --gguf -y
lms load qwen/qwen3-4b-2507 \
  --identifier qwen3-4b-instruct-2507 \
  --gpu max \
  --context-length 4096 \
  -y
```

## Local Requirements

Required for the default path:

- macOS microphone permission for the terminal app.
- `whisper-cli` from `whisper.cpp`.
- A Whisper model path, or a `whisper-cli` default model.
- LM Studio running a local OpenAI-compatible server.
- A local chat model loaded in LM Studio.
- A short voice reference WAV for Chatterbox Turbo. It does not need to be your
  voice.

Reasoning-heavy models may spend part of the generation budget before returning
the final question, so the default follow-up budget is intentionally larger
than a short answer appears to need. For faster follow-ups, use a smaller
non-thinking model and set `--lmstudio-max-tokens` lower.

Check the environment:

```sh
uv run interview-agent doctor
uv run interview-agent doctor --json
uv run --extra live interview-agent devices
```

Before copying a `whisper.cpp` command, set `WHISPER_MODEL` to a real local ggml
model file:

```sh
export WHISPER_MODEL="$HOME/.local/share/transcribe-audio/models/ggml-base.en.bin"
test -f "$WHISPER_MODEL"
```

## Voice Reference

Chatterbox Turbo is a voice-cloning TTS model. A voice reference is a short WAV
clip that tells the model what voice to synthesize.

It does not need to be the user's voice. For a neutral local reference generated
with macOS `say`:

```sh
uv run interview-agent make-reference ./voice-reference.wav
```

Record one:

```sh
uv run --extra live interview-agent record-reference ./voice-reference.wav \
  --input-device "MacBook Pro Microphone"
```

Use a clean 10-second clip with normal speaking volume. The reference is local;
do not commit it to the repo.

## Quick Start

Run with `whisper.cpp`, TEN VAD, Chatterbox Turbo, and LM Studio:

```sh
uv run --extra live interview-agent run \
  --voice-reference ./voice-reference.wav \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL" \
  --initial-question "What idea should we turn into video raw material?"
```

Say `next question` when the answer is complete. Long pauses are allowed.

Run with Kokoro instead of Chatterbox:

```sh
uv run --extra live interview-agent run \
  --tts kokoro \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL" \
  --max-turns 2
```

Kokoro downloads the int8 ONNX model and voices file on first use, then reuses
them from `.cache/`.

Run with Supertonic for a faster local voice. Local smoke tests showed
Supertonic can occasionally drop or mangle words in short prompts, so use Piper
when prompt intelligibility matters more than tone:

```sh
uv run --extra live interview-agent run \
  --tts supertonic \
  --tts-num-threads 4 \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL" \
  --max-turns 2
```

Supertonic includes multiple local speakers. Speaker `0` is the default because
it was the most reliable Supertonic speaker in local short-prompt smoke tests.
Other speaker IDs may sound different, but several dropped words from the start
of short questions.

```sh
uv run --extra live interview-agent run \
  --tts supertonic \
  --tts-speaker-id 6 \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL" \
  --max-turns 2
```

Run with Piper when the lowest prompt latency matters most:

```sh
uv run --extra live interview-agent run \
  --tts piper \
  --tts-num-threads 4 \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL" \
  --max-turns 2
```

For the fastest interactive path, use a small control model for detecting the
explicit done phrase and a separate model for the full answer transcript:

```sh
export WHISPER_CONTROL_MODEL="$HOME/.local/share/transcribe-audio/models/ggml-tiny.en.bin"

uv run --extra live interview-agent run \
  --tts kokoro \
  --input-device "MacBook Pro Microphone" \
  --whisper-control-model "$WHISPER_CONTROL_MODEL" \
  --whisper-model "$WHISPER_MODEL" \
  --lmstudio-model qwen3-4b-instruct-2507 \
  --lmstudio-max-tokens 120 \
  --max-turns 2
```

If `tiny.en` misses the done phrase in your room/noise setup, use `base.en` for
both `--whisper-control-model` and `--whisper-model`.

For lower turn-transition latency, use Moonshine streaming ASR. This keeps the
ASR model loaded, streams microphone audio while you speak, and uses the live
transcript immediately after the explicit done phrase instead of blocking on a
separate full-answer Whisper transcription:

```sh
uv run --extra live interview-agent run \
  --stt moonshine_streaming \
  --moonshine-model small_streaming \
  --tts supertonic \
  --input-device "MacBook Pro Microphone" \
  --done-phrase next \
  --done-phrase "next slide" \
  --lmstudio-model qwen3-4b-instruct-2507 \
  --lmstudio-max-tokens 120 \
  --timings \
  --max-turns 2
```

Add `--stream-transcripts` when you want to debug the evolving Moonshine text
while you speak. Normal runs keep that output quiet and print only turn status
plus timing lines.

On Chatterbox startup you may see Hugging Face print `Fetching 10 files` even
after the model has already been downloaded. If it says `Download complete:
0.00B`, it is checking cached files, not downloading gigabytes again. The
remaining delay is model load plus speech synthesis. Generated Chatterbox audio
is cached under `.cache/` so repeated identical questions can replay without
loading the model again.

For a lower-friction smoke test without Chatterbox:

```sh
uv run --extra live interview-agent run \
  --tts macos_say \
  --followup static \
  --input-device "MacBook Pro Microphone" \
  --whisper-model "$WHISPER_MODEL"
```

Use the sherpa-onnx STT adapter:

```sh
uv run --extra live interview-agent run \
  --stt sherpa_onnx \
  --sherpa-model-dir /path/to/sherpa/model \
  --sherpa-model-kind auto \
  --tts macos_say \
  --followup static
```

The sherpa adapter supports common offline sherpa layouts: transducer
`encoder`/`decoder`/`joiner`, sherpa Whisper `encoder`/`decoder`, Paraformer,
NeMo CTC, and WeNet CTC. Use `--sherpa-model-kind` when `auto` cannot infer the
layout from filenames.

Use local model directories for Piper or Supertonic when you do not want the
defaults downloaded into `.cache/`:

```sh
uv run --extra live interview-agent run \
  --tts supertonic \
  --supertonic-model-dir /path/to/sherpa-onnx-supertonic-tts-int8 \
  --followup static
```

## Commands

| Command | Purpose |
| --- | --- |
| `interview-agent doctor` | Check installed tools and optional packages |
| `interview-agent devices` | List Core Audio input and output devices |
| `interview-agent record-reference OUT.wav` | Record a Chatterbox Turbo voice reference |
| `interview-agent make-reference OUT.wav` | Generate a neutral local voice reference with macOS `say` |
| `interview-agent run` | Start a live microphone interview session |
| `interview-agent ask-followup transcript.txt` | Ask LM Studio for one follow-up |

## Outputs

Each session writes a timestamped directory:

```text
sessions/
  20260704-181530/
    answer-001.wav
    answer-001.partial.wav
    answer-001.tail.wav
    session.json
```

`session.json` records questions, answer audio paths, control transcripts, final
transcripts, and the done phrase that stopped each turn.

## Backend Boundaries

The project keeps backends behind small interfaces:

```python
class VADBackend:
    def speech_segments(self, path): ...

class STTBackend:
    def transcribe_file(self, path): ...

class TTSBackend:
    def speak(self, text): ...

class FollowupBackend:
    def next_question(self, transcript_so_far): ...
```

That makes it possible to swap `whisper.cpp` and `sherpa-onnx` without changing
the session loop.

## Privacy

No background recording. The microphone is used only during an explicit
`interview-agent run` session.

Raw audio and transcripts are written locally. The default follow-up backend
sends transcript text to LM Studio on `localhost`; it does not send audio to a
remote API.

## Development

```sh
uv run --extra dev pytest
uv run --extra dev ruff check .
```

Keep generated `sessions/` data, voice references, and local model paths out of
commits.
