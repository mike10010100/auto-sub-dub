# Auto-Dub

A fully local, open-source pipeline for automatic video dubbing using AI.

## Features
- **Vocal Isolation:** Separates vocals from background music/effects using Meta's Demucs.
- **Transcription & Diarization:** High-precision transcription and speaker identification using WhisperX.
- **LLM Translation:** Context-aware translation using Gemma 4 via a local Ollama instance.
- **Expressive Voice Cloning:** Zero-shot voice cloning to match original character voices.
- **Professional Mixing:** Re-mixes dubbed voices with original background audio and remuxes into the final video.

## Prerequisites
1. **FFmpeg:** Ensure `ffmpeg` is installed on your system.
2. **Ollama:** Ensure Ollama is running with the `gemma4` model pulled.
   - Default endpoint: `http://192.168.86.172:11434`
3. **Hugging Face Token:** Required for speaker diarization (WhisperX uses `pyannote/speaker-diarization-3.1`).
   - Accept terms at: [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - Accept terms at: [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

## Installation & Tool Management

This project uses [**mise-en-place**](https://mise.jdx.dev/) for seamless management of Python, FFmpeg, and project tasks. 

If you haven't already, install `mise`:
```bash
curl https://mise.run | sh
```

Once `mise` is installed, set up your environment:
```bash
# Installs Python 3.11 and the latest FFmpeg automatically
mise install

# Install project dependencies
mise run install
```

## Usage
Run the pipeline using `mise`:
```bash
mise run run -- video.mp4 --lang "Spanish" --hf_token "YOUR_TOKEN"
```


## Hardware Optimization
Since you have dual GPUs:
- **RTX 3500 Ada (12GB):** Best for the TTS / Voice Cloning engine.
- **RTX 3080 (10GB):** Best for WhisperX and Demucs.
- You can specify the device in the code by modifying the `device` parameters in `main.py`.

## Pipeline Stages
1. **Audio Extraction:** Extract original audio from MP4.
2. **Separation:** Split into `vocals.wav` and `background.wav`.
3. **Transcription:** Diarized transcript (who said what and when).
4. **Translation:** Gemma 4 translates the text while maintaining tone.
5. **Synthesis:** Clone voices from original clips and generate dubbed segments.
6. **Assembly:** Overlay segments on background audio and remux to MP4.
