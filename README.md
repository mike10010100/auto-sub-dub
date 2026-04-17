# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI. Now with support for **NVIDIA (CUDA)** and **Apple Silicon (MPS)**.

## Features
- **Cross-Platform Acceleration:** Automatically detects and uses CUDA (NVIDIA) or MPS (Apple Silicon) for the heaviest tasks (Transcription, Synthesis).
- **Vocal Isolation:** Separates vocals from background music/effects using Meta's **Demucs**.
- **Transcription & Diarization:** High-precision transcription and speaker identification using **WhisperX**.
- **Hybrid Multimodal Translation:** Context-aware and **emotion-aware** translation using **Gemma 4 (26B)** via a local Ollama instance. The model "listens" to the original audio to capture sarcasm and tone.
- **Expressive Voice Cloning:** Zero-shot voice cloning using **Coqui XTTS v2** with **Multi-Reference Triangulation** (3+ samples per speaker) and emotional conditioning.
- **Professional Mixing:** Re-mixes dubbed voices with original background audio using **Vocal Ducking** (sidechain compression) to keep speech clear.
- **Resiliency & Checkpointing:** Smart pipeline that skips already-completed steps if you re-run a job.

## Prerequisites
1. **FFmpeg:** Ensure `ffmpeg` is installed on your system.
2. **Ollama:** Ensure Ollama is running with the `gemma4` model pulled.
3. **macOS Build Tools:** If you are on a Mac, install the command line tools for better performance:
   ```bash
   xcode-select --install
   ```
4. **Environment Setup:** 
   - Copy `.env.example` to `.env`.
   - Add your **Hugging Face Token** (required for speaker diarization).
   - Configure your **Ollama URL** and **Model** if they differ from defaults.

## Installation & Tool Management

This project uses [**mise-en-place**](https://mise.jdx.dev/) for seamless management of Python and project tasks.

```bash
# 1. Install mise (if not already installed)
curl https://mise.run | sh

# 2. Install Python 3.11 automatically
mise install

# 3. Install project dependencies
mise run install
```

## Usage
### Web Interface (Recommended)
Run the Streamlit dashboard for a user-friendly experience:
```bash
mise run web
```
This allows for easy video uploads, device selection (CUDA/MPS/CPU), and progress monitoring.

### Command Line
Run the pipeline using `mise`:
```bash
# Basic usage
mise run run -- your_video.mp4 --lang "Spanish"

# Specify a device
mise run run -- your_video.mp4 --lang "French" --device mps
```

The dubbed output will be saved in a per-video project folder in the `output/` directory.

## Hardware Requirements
- **GPU:** 
  - **NVIDIA:** 10GB+ VRAM recommended (RTX 3080 or better).
  - **Apple Silicon:** M1 Pro or better recommended (16GB+ Unified Memory).
- **Disk Space:** ~10-15GB for model weights.

## Project Structure
- `main.py`: Orchestrator and entry point.
- `app.py`: Streamlit web dashboard.
- `src/utils.py`: Hardware detection and compute utilities.
- `src/audio_processor.py`: Audio extraction, vocal isolation, and ducking.
- `src/transcriber.py`: Transcription and speaker diarization.
- `src/translator.py`: Multimodal translation via Ollama.
- `src/synthesizer.py`: Voice cloning and speed adjustment.
