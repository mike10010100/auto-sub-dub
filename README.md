# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI. Now with support for **NVIDIA (CUDA)** and **Apple Silicon (MPS)**.

## Features
- **Cross-Platform Acceleration:** Automatically detects and uses CUDA (NVIDIA) or MPS (Apple Silicon).
    - **Note:** For maximum quality and compatibility, the pipeline uses **CPU** for Synthesis on Mac, while keeping isolation and transcription on the GPU where possible.
- **Vocal Isolation:** Separates vocals from background music/effects using Meta's **Demucs**, on two parallel tracks — a 16 kHz mono track tuned for ASR and a 44.1 kHz stereo track tuned for voice cloning.
- **Transcription & Diarization:** High-precision transcription and speaker identification using **WhisperX**.
- **Hybrid Two-Model Translation:** Audio-informed emotion tagging with **`gemma4:e4b`** (Gemma 4's audio-capable E-variant, fed the raw segment WAV) in parallel, then sequential **`gemma4:26b`** text translation with a rolling ±5/+3 segment context window. Disambiguates greetings, pronouns, and sarcasm that per-segment translation gets wrong.
- **Emotion-Conditioned Voice Cloning:** Zero-shot cloning with **Coqui XTTS v2** using **Multi-Reference Triangulation**. References are drawn from the 44.1 kHz stereo vocals track and ranked by **Silero VAD** voiced-frame ratio + SNR (not just loudness), so music-heavy or low-voice clips don't poison the clone. Each segment's synthesis picks the reference whose source emotion matches the target line — XTTS's own `emotion=` kwarg is a no-op, but reference-clip affect carries through reliably.
- **Reference Resilience:** Every diarized speaker gets a reference. Long monologue blobs are trimmed to the XTTS sweet spot instead of being rejected, and a fallback takes the longest available clip if no candidate passes quality filters — so side characters never go silent in the dub.
- **Natural-Pace Timing:** Synthesized dub is placed into a per-segment window widened by stealing silence from adjacent gaps (up to ±1 s). WSOLA time-stretching only ever speeds audio up (1.0–2.0×), never slowing it below XTTS's natural pace.
- **Professional Mixing:** Re-mixes dubbed voices with original background using smoothed **sidechain ducking** (RMS envelope, asymmetric 50 ms / 400 ms attack/release) — no pumping.
- **Resiliency & Checkpointing:** Per-stage artifact caching (audio extract, Demucs separation, transcript, translated transcript, references, synthesized segments) — re-runs skip anything already on disk.

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

### Command Line
Run the pipeline using `mise`:
```bash
# Basic usage
mise run run -- your_video.mp4 --lang "Spanish"

# Specify a device
mise run run -- your_video.mp4 --lang "French" --device mps

# Override models (rare — env vars from mise.toml / .env cover the defaults)
mise run run -- your_video.mp4 --lang "English" \
  --ollama_model gemma4:26b --ollama_audio_model gemma4:e4b
```

The dubbed output will be saved in a per-video project folder in the `output/` directory.

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
ustment.
lation, and ducking.
- `src/transcriber.py`: Transcription and speaker diarization.
- `src/translator.py`: Multimodal translation via Ollama.
- `src/synthesizer.py`: Voice cloning and speed adjustment.
