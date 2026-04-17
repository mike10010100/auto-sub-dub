# Instructions for AI Agents

Welcome, Agent. You are assisting in the development and maintenance of **Auto-Dub**, a fully local AI-powered video dubbing pipeline.

## Project Architecture
Auto-Dub is a modular Python application that performs the following steps:
1.  **Vocal Isolation:** Uses Meta's **Demucs** to separate vocals from background music/noise.
2.  **Transcription & Diarization:** Uses **WhisperX** (Whisper + Pyannote.audio) to generate millisecond-accurate timestamps and speaker identities.
3.  **Hybrid Multimodal Translation:** Calls a remote **Ollama** instance (running **Gemma 4 26B**). The agent sends the raw audio of each segment + transcript text to Gemma 4 to capture emotional prosody.
4.  **Voice Cloning & Synthesis:** Uses **Coqui XTTS v2** (native Python library). It uses **Multi-Reference Triangulation** (3+ 6-12s clips) and **Emotion Conditioning** tags to match the original performance.
5.  **Assembly:** Uses **pydub** for WSOLA time-stretching/mixing and **FFmpeg** for final remuxing.

## Development Environment
- **Tool Manager:** [mise-en-place](https://mise.jdx.dev/). Use `mise run <task>` for common operations.
- **Python Version:** 3.11 (managed by mise).
- **GPU Targeting:** The pipeline is designed for NVIDIA GPUs (10GB+ VRAM). It currently uses `cuda` for both WhisperX and XTTS v2.

## Key Files
- `main.py`: The entry point and orchestrator.
- `src/translator.py`: Handles the Multimodal API calls to Ollama.
- `src/synthesizer.py`: Handles multi-reference extraction and conditional synthesis.
- `mise.toml`: Defines tools, tasks, and environment variables (including `COQUI_TOS_AGREED=1`).
- `.env`: Stores sensitive credentials like `HF_TOKEN`. **Never commit this file.**
- `PRD.md`: The single source of truth for project goals.

## Operational Guidelines
- **Resiliency:** The pipeline supports checkpointing. It skips steps if it finds existing files in the `output/<video_name>/` directory.
- **Thinking Mode:** When calling Gemma 4, always include `<|think|>` at the start of the system prompt and parse out the `<|channel>thought` blocks.
- **Dependencies:** Always check `requirements.txt` for pinned versions (especially `torch`, `torchaudio`, and `ollama`).

## Task Execution
- `mise run install`: Installs all dependencies.
- `mise run run -- <video.mp4> --lang <Language>`: Executes the full pipeline.
