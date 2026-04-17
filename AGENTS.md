# Instructions for AI Agents

Welcome, Agent. You are assisting in the development and maintenance of **Auto-Dub**, a fully local AI-powered video dubbing pipeline.

## Project Architecture
Auto-Dub is a modular Python application that performs the following steps:
1.  **Vocal Isolation:** Uses Meta's **Demucs** to separate vocals from background music/noise.
2.  **Transcription & Diarization:** Uses **WhisperX** (Whisper + Pyannote.audio) to generate timestamped text with speaker identities.
3.  **Translation:** Calls a remote **Ollama** instance (running **Gemma 4**) via API.
4.  **Voice Cloning & Synthesis:** Uses **Coqui XTTS v2** (native Python library) to clone original speakers' voices and generate the dubbed audio.
5.  **Assembly:** Uses **pydub** for time-stretching/mixing and **FFmpeg** for final remuxing.

## Development Environment
- **Tool Manager:** [mise-en-place](https://mise.jdx.dev/). Use `mise run <task>` for common operations.
- **Python Version:** 3.11 (managed by mise).
- **GPU Targeting:** The pipeline is designed for NVIDIA GPUs (10GB+ VRAM). It currently uses `cuda` for both WhisperX and XTTS v2.

## Key Files
- `main.py`: The entry point and orchestrator.
- `src/`: Contains modular logic for each phase of the pipeline.
- `mise.toml`: Defines tools, tasks, and environment variables.
- `.env`: Stores sensitive credentials like `HF_TOKEN`. **Never commit this file.**
- `AGENTS.md`: This file.

## Operational Guidelines
- **Resiliency:** The pipeline supports checkpointing. It skips steps if it finds existing files in the `output/` directory (e.g., `transcript.json`, `vocals.wav`).
- **Dependencies:** If adding new AI models, prioritize libraries that can run locally on Linux/WSL2 with CUDA support.
- **Tooling:** Always check `requirements.txt` for pinned versions (especially `torch` and `torchaudio`) to avoid API breaking changes.

## Task Execution
- `mise run install`: Installs all dependencies.
- `mise run run -- <video.mp4> --lang <Language>`: Executes the full pipeline.
- `mise run test`: Runs a test of the synthesizer logic.

## Remote Context
- **Ollama Endpoint:** `http://192.168.86.172:11434` (Default). This is the only remote dependency.
- **Hugging Face:** Requires `HF_TOKEN` for gated speaker diarization models.
