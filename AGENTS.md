# Instructions for AI Agents

Welcome, Agent. You are assisting in the development and maintenance of **Auto-Dub**, a fully local AI-powered video dubbing pipeline.

## Project Architecture
Auto-Dub is a modular Python application that performs the following steps:
1.  **Vocal Isolation:** Uses Meta's **Demucs** to separate vocals from background music/noise.
2.  **Transcription & Diarization:** Uses **WhisperX** (Whisper + Pyannote.audio). Includes a global monkey-patch for `hf_hub_download` to handle the `use_auth_token` vs `token` rename in recent libraries.
3.  **Hybrid Multimodal Translation:** Calls a local/remote **Ollama** instance (running **Gemma 4 26B**). Uses `ThreadPoolExecutor` for concurrent segment translation.
4.  **Voice Cloning & Synthesis:** Uses **Coqui XTTS v2**. Implements **Multi-Reference Triangulation** (3+ clips per speaker) and **Emotion Conditioning**.
5.  **Assembly:** Uses **pydub** for WSOLA time-stretching, **Vocal Ducking** (sidechain compression), and **FFmpeg** for final remuxing.

## Development Environment
- **Tool Manager:** [mise-en-place](https://mise.jdx.dev/).
- **Python Version:** 3.11.
- **Hardware Agnostic:** Uses `src/utils.py` for device detection.
    - **NVIDIA:** Uses `cuda` (preferred).
    - **Apple Silicon:** Uses `mps` for Isolation, but falls back to `cpu` for Synthesis (quality/stability) and Transcription (`ctranslate2` limitation).
- **Environment:** Uses `python-dotenv` for `.env` loading.

## Key Files
- `main.py`: The orchestrator and CLI entry point.
- `app.py`: The Streamlit web dashboard.
- `src/utils.py`: Device detection and hardware utilities.
- `src/translator.py`: Concurrent multimodal API calls to Ollama.
- `src/synthesizer.py`: Multi-reference extraction and conditional synthesis.
- `src/transcriber.py`: High-precision transcription with Diarization monkey-patches.

## Operational Guidelines
- **Resiliency:** Supports checkpointing. Skips steps if existing artifacts are found.
- **Logging:** Use `logging.getLogger(__name__)`. Silence `speechbrain` noise in UI contexts.
- **Thinking Mode:** Include `<|think|>` in Gemma 4 prompts and parse the JSON output from the final block.
- **Mac Stability:** For XTTS v2 on ARM64, always force `cpu` to avoid hallucinations.

## Task Execution
- `mise run install`: Installs all dependencies.
- `mise run web`: Launches the Streamlit dashboard.
- `mise run run -- <video.mp4> --lang <Language>`: Executes the CLI pipeline.
