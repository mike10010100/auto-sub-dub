# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI.

## Features
- **Vocal Isolation:** Separates vocals from background music/effects using Meta's **Demucs**.
- **Transcription & Diarization:** High-precision transcription and speaker identification using **WhisperX**.
- **LLM Translation:** Context-aware translation using **Gemma 4** via a local Ollama instance.
- **Expressive Voice Cloning:** Zero-shot voice cloning using **Coqui XTTS v2** to match original character voices.
- **Resiliency & Checkpointing:** Smart pipeline that skips already-completed steps (transcription, translation, synthesis) if you re-run a job.
- **Professional Mixing:** Re-mixes dubbed voices with original background audio and remuxes into the final video with automatic time-stretching.

## Prerequisites
1. **FFmpeg:** Ensure `ffmpeg` is installed on your system (managed by `mise` or system package manager).
2. **Ollama:** Ensure Ollama is running with the `gemma4` model pulled.
   - Default endpoint: `http://192.168.86.172:11434`
3. **Hugging Face Token:** Required for speaker diarization models.
   - Accept terms at: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Accept terms at: [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - Add your token to a `.env` file: `HF_TOKEN=your_token_here`

## Installation & Tool Management

This project uses [**mise-en-place**](https://mise.jdx.dev/) for seamless management of Python and project tasks.

```bash
# 1. Install mise (if not already installed)
curl https://mise.run | sh

# 2. Install Python 3.11 automatically
mise install

# 3. Install project dependencies (AI models, etc.)
mise run install
```

## Usage
Run the pipeline using `mise`:
```bash
# Use a local .mp4 file
mise run run -- your_video.mp4 --lang "English"
```

The dubbed output will be saved in the `output/` directory.

## Hardware Requirements
- **GPU:** NVIDIA GPU with at least **10GB+ VRAM** is recommended (tested on RTX 3080 10GB).
- **Disk Space:** ~5-10GB for model weights (Whisper, Demucs, XTTS v2).

## Project Structure
- `main.py`: Orchestrator and entry point.
- `src/audio_processor.py`: Audio extraction and vocal isolation.
- `src/transcriber.py`: Transcription and speaker diarization.
- `src/translator.py`: Ollama/Gemma 4 translation logic.
- `src/synthesizer.py`: Voice cloning and time-stretching.
- `AGENTS.md`: Detailed architecture guide for AI developers.
