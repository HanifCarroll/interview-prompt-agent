from interview_prompt_agent.stt.whisper_cpp import _clean_output


def test_clean_output_strips_whisper_cpp_noise() -> None:
    output = """
load_backend: loaded CPU backend
ggml_metal_device_init: GPU name: MTL0
read_audio_data: reading audio data

  The actual transcript.
"""
    assert _clean_output(output) == "The actual transcript."
