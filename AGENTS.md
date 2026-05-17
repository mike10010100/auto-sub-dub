# Instructions for AI Agents

Welcome, Agent. You are assisting in the development and maintenance of **Auto-Dub**, a fully local AI-powered video dubbing pipeline.

## Project Architecture
Auto-Dub is a modular Python application that performs the following steps:
1.  **Vocal Isolation:** Uses Meta's **Demucs** to separate vocals from background music/noise.
2.  **Transcription & Diarization:** Uses **WhisperX** (Whisper + Pyannote.audio). Includes a global monkey-patch for `hf_hub_download` to handle the `use_auth_token` vs `token` rename in recent libraries.
3.  **Hybrid Multimodal Translation:** Calls a local/remote **Ollama** instance (running **Gemma 4 26B**). Uses sequential segment translation with rolling context.
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
- `mise run install`: Installs runtime dependencies.
- `mise run install-dev`: Installs runtime + dev (ruff, pytest, pytest-cov) deps.
- `mise run web`: Launches the Streamlit dashboard.
- `mise run run -- <video.mp4> --lang <Language>`: Executes the CLI pipeline.
- `mise run format`: Auto-format with ruff.
- `mise run lint` / `mise run lint-fix`: Lint with ruff.
- `mise run unit`: Run unit tests with coverage.
- **`mise run is-valid`**: Format-check + lint + unit tests w/ 90% coverage gate. **REQUIRED before declaring any change done.**
- `mise run integration`: End-to-end pipeline test against real models + Ollama. Optional — requires GPU/CPU headroom, a running Ollama with `gemma4:26b` + `gemma4:e4b`, a valid `HF_TOKEN`, and `sample_video.mp4` at the repo root.

## Quality Gate (mandatory for agents)

Before reporting any code change as complete, run `mise run is-valid` and fix everything it flags. It runs in order:

1. `ruff format --check .` — formatting must be idempotent.
2. `ruff check .` — lint rules (E, F, W, I, B, UP, C4, SIM) must pass.
3. `pytest --cov --cov-fail-under=90` — unit tests must pass and branch coverage across the measured source (`src/utils.py`, `src/audio_processor.py`, `src/translator.py`, `src/timing.py`) must stay ≥ 90%.

Coverage is deliberately scoped to pure-logic modules. Model-bound IO (`src/synthesizer.py`, `src/transcriber.py`, and method bodies marked `# pragma: no cover`) is exercised by `mise run integration`, not by the unit suite. When you add new pure-logic code, add tests for it — do not rely on integration to cover it.

If you add a new pure helper, place it somewhere the coverage source list already measures (`src/timing.py`, `src/utils.py`, or a top-level function in `audio_processor.py` / `translator.py`). If you add a new module that needs coverage, include it in `[tool.coverage.run] source` in `pyproject.toml`.

### Style notes
- Lazy-import `torch`, `whisperx`, `TTS`, and other heavy deps inside functions rather than at module top, so test collection stays fast and coverage tracing doesn't clash with C extensions.
- Mark IO / model-bound methods with `# pragma: no cover` and ensure the integration test exercises them.
