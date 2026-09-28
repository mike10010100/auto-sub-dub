# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI. Now with support for **NVIDIA (CUDA)** and **Apple Silicon (MPS)**.

## Features
- **Cross-Platform Acceleration:** Automatically detects and uses CUDA (NVIDIA) or MPS (Apple Silicon), with optional headless remote offloading to **ComfyUI** and **Ollama**.
- **Studio Vocal Isolation:** Separates vocals from background audio using **Mel-Band Roformer** (via ComfyUI) or Meta's **Demucs** fallback, providing clean, artifact-free vocal stems for downstream voice cloning.
- **Transcription & Diarization:** High-precision transcription and speaker identification using **WhisperX** (with support for `faster-whisper-large-v3-turbo` and Pyannote 3.1).
- **Hybrid Two-Model Translation:** Audio-informed emotion tagging with **`gemma4:e4b`** in parallel, followed by sequential **`gemma4:26b`** (or `gemma4:31b`) text translation with a rolling segment context window and syllable/duration budgeting.
- **Flow-Matching Zero-Shot Voice Cloning (F5-TTS & CosyVoice):** Uses non-autoregressive Flow Matching (F5-TTS) via ComfyUI for zero-hallucination, naturally paced voice cloning matching the original actor's timbre without accent contamination. Also supports local **Fish Speech + Seed-VC** or legacy **XTTS v2**.
- **Visual Lip-Syncing (LatentSync):** Optional end-to-end latent diffusion lip synchronization (ByteDance **LatentSync** via ComfyUI), re-animating the speaker's mouth to match the dubbed track with high-resolution temporal consistency.
- **Natural-Pace Timing & Ducking:** WSOLA time-stretching with gap stealing, loudness normalization, and smooth sidechain ducking.
- **Resiliency & Checkpointing:** Per-stage artifact caching (audio extract, separation, transcript, translated transcript, references, synthesized segments, lip-sync).

## Prerequisites
1. **FFmpeg:** Ensure `ffmpeg` is installed on your system.
2. **Ollama:** Ensure Ollama is running with **both** models pulled — `gemma4:26b` (text translation) and `gemma4:e4b` (audio-informed emotion tagging; must be an E-variant for audio support).
3. **macOS Build Tools:** If you are on a Mac, install the command line tools for better performance:
   ```bash
   xcode-select --install
   ```
4. **Environment Setup:** 
   - Copy `.env.example` to `.env`.
   - Add your **Hugging Face Token** (required for speaker diarization).
   - Configure your **Ollama URL**, **translation model** (`OLLAMA_MODEL`, e.g. `gemma4:26b`), and **audio model** (`OLLAMA_AUDIO_MODEL`, e.g. `gemma4:e4b`) if they differ from defaults.

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

### Docker Deployment (One-Step)
If you are deploying to a server with NVIDIA GPUs (like Lambda Labs), you can set up the entire stack with one command:

1. **Clone the repo:**
   ```bash
   git clone https://github.com/mike10010100/auto-sub-dub.git
   cd auto-sub-dub
   ```
2. **Launch the stack:**
   ```bash
   # Set your HF token first
   export HF_TOKEN=your_token_here
   docker-compose up --build
   ```

This will automatically:
* Start an Ollama server and pull the `gemma4:26b` and `gemma4:e4b` models.
* Build the application container with CUDA support.
* Compile the `s2.cpp` high-fidelity audio engine.
* Download the Fish Speech GGUF models.
* Launch the Streamlit web interface on port `8501`.

### Command Line
Run the pipeline using `mise`:
```bash
# High-fidelity run with ComfyUI (Mel-Band Roformer + F5-TTS + LatentSync lip-syncing)
mise run run -- your_video.mp4 --lang "Spanish" --engine f5 --separator roformer --lipsync

# Fast dub without visual lip-sync
mise run run -- your_video.mp4 --lang "English" --engine f5 --separator roformer

# Using local Fish Speech + Seed-VC with MPS/CUDA
mise run run -- your_video.mp4 --lang "Spanish" --engine fish --device mps
```

The dubbed output will be saved in a per-video project folder in the `output/` directory.

### Zero-Accent Voice Conversion (Seed-VC)
Zero-shot cloning (even with Fish Speech) can sometimes carry over the original speaker's phonetic accent (e.g., a Japanese accent when speaking English). To completely eliminate this and achieve 100% character voice fidelity, Auto-Dub now has **Seed-VC** fully integrated.

**How it works automatically:**
1. Auto-Dub asks Fish Speech to generate flawless, natively accented English audio (acting as a generic voice actor).
2. The pipeline then automatically passes that pristine English audio through the Seed-VC zero-shot engine, alongside the perfectly isolated Japanese reference clip it extracted earlier.
3. Seed-VC instantly "re-skins" the English audio with the character's exact pitch and timbre. The result is perfect acting with zero accent leakage, requiring **zero manual model training.**

## Development

Install dev tools (ruff, pytest, pytest-cov):

```bash
mise run install-dev
```

Before committing any change, run the validation gate:

```bash
mise run is-valid   # format-check + lint + unit tests with 90% coverage
```

Optional end-to-end test (requires models, Ollama, HF token, and `sample_video.mp4`):

```bash
mise run integration
```

See `AGENTS.md` for contributor guidelines and the quality-gate contract enforced on AI agent work.

## Hardware Requirements
- **GPU:** 
  - **NVIDIA:** 10GB+ VRAM recommended (RTX 3080 or better). 10GB is sufficient for Fish Speech S2 Pro via the 4-bit GGUF engine.
  - **Apple Silicon:** M1 Pro or better recommended (16GB+ Unified Memory).
- **Disk Space:** ~10-15GB for model weights (+4GB if using Fish Speech).

## Troubleshooting
- **Shutdown Errors (RuntimeError):** On some macOS systems, you may see a `RuntimeError: reentrant call` when closing the Streamlit dashboard with `Ctrl+C`. This is a known issue with Streamlit's internal signal handling and does not affect the correctness of your dubbing projects.
- **Diarization Fails:** Ensure your `HF_TOKEN` is valid and you have accepted the terms for the `pyannote/speaker-diarization` and `pyannote/segmentation` models on Hugging Face.
- **Ollama Connection:** Verify Ollama is running and accessible at the URL provided in the UI or `.env`.

## Project Structure
- `main.py`: Orchestrator and entry point.
- `app.py`: Streamlit web dashboard.
- `src/utils.py`: Hardware detection and compute utilities.
- `src/audio_processor.py`: Audio extraction, vocal isolation, and ducking.
- `src/transcriber.py`: Transcription and speaker diarization.
- `src/translator.py`: Multimodal translation via Ollama.
- `src/synthesizer.py`: Voice cloning and speed adjustment.
