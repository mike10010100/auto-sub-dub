# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI.

## Features
- **Vocal Isolation:** Separates vocals from background music/effects using Meta's **Demucs**.
- **Transcription & Diarization:** High-precision transcription and speaker identification using **WhisperX**.
- **Hybrid Multimodal Translation:** Context-aware and **emotion-aware** translation using **Gemma 4 (26B)** via a local Ollama instance. The model "listens" to the original audio to capture sarcasm and tone.
- **Expressive Voice Cloning:** Zero-shot voice cloning using **Coqui XTTS v2** with **Multi-Reference Triangulation** (3+ samples per speaker) and emotional conditioning.
- **Resiliency & Checkpointing:** Smart pipeline that skips already-completed steps (transcription, translation, synthesis) if you re-run a job.
- **Professional Mixing:** Re-mixes dubbed voices with original background audio and remuxes into the final video with automatic time-stretching.

## Prerequisites
1. **FFmpeg:** Ensure `ffmpeg` is installed on your system (managed by system package manager).
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

The dubbed output will be saved in a per-video project folder in the `output/` directory.

## Hardware Requirements
- **GPU:** NVIDIA GPU with at least **10GB+ VRAM** is recommended (tested on RTX 3080 10GB).
- **Disk Space:** ~10-15GB for model weights (Whisper, Demucs, XTTS v2, Gemma 4).

## Project Structure
- `main.py`: Orchestrator and entry point.
- `src/audio_processor.py`: Audio extraction and vocal isolation.
- `src/transcriber.py`: Transcription and speaker diarization.
- `src/translator.py`: **Multimodal** Ollama/Gemma 4 translation and emotion detection.
- `src/synthesizer.py`: Voice cloning, triangulation, and time-stretching.
- `AGENTS.md`: Detailed architecture guide for AI developers.
- `PRD.md`: Full project roadmap and requirements.
