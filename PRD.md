# Auto-Dub PRD & Architecture Plan

## Objective
Build a fully local, open-source pipeline that takes a video file, isolates vocals, transcribes/diarizes the audio, translates the transcript using a local LLM (Gemma 4 via Ollama), generates expressive dubbed audio using voice cloning, and remuxes the audio back into the video.

## Key Files & Context
- **Repository:** `/home/mike10010100/git/auto-sub-dub`
- **LLM Endpoint:** `http://192.168.86.157:11434` (Ollama running Gemma 4 26B)
- **Hardware Profile:** NVIDIA RTX 3080 (10GB VRAM) for local processing, RTX 3500 Ada for Ollama.

## Architecture & Technology Stack
1. **Audio Extraction & Mixing:** `FFmpeg`
2. **Vocal Isolation:** `Demucs` - to separate vocals from background noise/music.
3. **Transcription & Diarization:** `WhisperX` - High-speed transcription with speaker diarization via `pyannote.audio`.
4. **Translation Engine:** **Gemma 4 26B** (via local Ollama API) - **Hybrid Multimodal** architecture. Uses both original text and raw segment audio to capture emotional context and prosody.
5. **Expressive Voice Cloning (TTS):** `Coqui XTTS v2` - Using **Multi-Reference Triangulation** (3+ clips per speaker) and **Emotion Conditioning** (tags like `[SARCASM]`, `[FRIENDLY]`) for high fidelity.
6. **Audio Stitching & Time-Stretching:** `pydub` and `audiotsm` (WSOLA) - For precise timestamp placement and speed adjustment.

## Implementation Phases

### Phase 1: Setup and Basic Extraction (Completed)
1. Initialize Python project structure and `requirements.txt`.
2. Implement `AudioProcessor` for audio extraction and `Demucs` vocal isolation.

### Phase 2: Transcription and Translation (Completed)
1. Implement `Transcriber` using `WhisperX` for diarized transcripts.
2. Implement **Multimodal `Translator`** with duration-aware prompting and audio-aware emotion detection using Gemma 4.

### Phase 3: Voice Cloning & Audio Synthesis (Completed)
1. Implement `Synthesizer` with multi-reference triangulation (XTTS v2).
2. Integrate emotional conditioning to match the performance detected by the multimodal LLM.
3. Apply WSOLA time-stretching to match original spoken durations.

### Phase 4: Assembly and Final Render (Completed)
1. Per-video project directories for output isolation.
2. Automatic overlay mixing and `FFmpeg` remuxing.

### Phase 5: Productionization (Completed)
1. **Sidechain Compression (Vocal Ducking):** Automatically lower background music volume when speech is active using `pydub`.
2. **Web-Based Interface:** Built a **Streamlit** dashboard for drag-and-drop video processing and manual configuration.
3. **Enterprise Logging:** Transitioned to structured logging and robust error handling across all modules.
4. **Parallel Processing:** Concurrent Ollama API calls for significantly faster translation.

### Phase 6: Cross-Platform & Stability (Completed)
1. **Apple Silicon Support:** Hardware-agnostic device detection for CUDA, MPS, and CPU.
2. **Mac-Specific Tuning:** Numerical stability fixes for XTTS v2 (CPU fallback) and WhisperX (float32 fallback).
3. **Dependency Hardening:** Finalized a "Gold" `requirements.txt` that works on both ARM64 and x86_64.
4. **Environment Awareness:** Explicit `.env` loading for seamless configuration.

### Phase 7: Translation & Synthesis Quality (Completed)
1. **Hybrid Two-Model Translation:** Audio-informed emotion tagging with `gemma4:e4b` (parallel) + sequential `gemma4:26b` text translation with a rolling ±5/+3 segment context window so pronouns, greetings, and callbacks resolve correctly.
2. **Dual-Track Audio Pipeline:** Parallel 16 kHz mono (ASR/diarization) and 44.1 kHz stereo (XTTS reference cloning) tracks, with Demucs separation on each. XTTS quality degraded noticeably on the ASR-optimized track.
3. **VAD + SNR Reference Ranking:** References are ranked by Silero VAD voiced-frame ratio + SNR estimate (not loudness), so music-heavy or low-voice clips don't poison the clone.
4. **Reference Resilience:** Every diarized speaker gets a reference. Long monologue blobs are trimmed to the XTTS sweet spot; a fallback takes the longest available clip if no candidate passes the quality filters. No more silent side characters.
5. **Natural-Pace Timing:** WSOLA time-stretching is floored at 1.0× (only speeds audio up, never below XTTS's natural pace). Placement windows steal silence from adjacent gaps (±1 s) to give TTS room without clamping.
6. **Loudness Matching (EBU R128 / LUFS):** `pyloudnorm` measures the isolated vocals' integrated loudness and normalizes the dubbed track to match before sidechain ducking (clipped at ±12 dB).
7. **Subtitle Reconciliation:** When the source video carries a target-language subtitle stream, extract via ffprobe/ffmpeg, align to diarized segments by timestamp overlap, and feed Gemma both the source line and the professional translation as a "strong hint, not gospel" reference. No-ops silently when no matching stream exists.

### Phase 8: Productionization / Developer Experience (Completed)
1. **Quality Gate (`mise run is-valid`):** `ruff format --check` + `ruff check` + `pytest --cov --cov-fail-under=90`. Mandatory before declaring any change done.
2. **Scoped Coverage:** Coverage is measured on pure-logic modules (`src/utils.py`, `src/audio_processor.py`, `src/translator.py`, `src/timing.py`); model-bound IO is excluded via `# pragma: no cover` and exercised by the integration test instead of fragile mocks.
3. **End-to-End Integration (`mise run integration`):** Optional real-model pipeline run on `sample_video.mp4`; verifies the final dubbed artifact has both video and audio streams.
4. **AGENTS.md Contract:** Quality-gate contract, style rules (lazy imports for heavy deps, pragma discipline), and guidance on where to place new pure helpers to keep coverage honest.

### Phase 9: Future Roadmap
1. **Text Normalization Before TTS:** Expand numbers, dates, ordinals, and common acronyms into spoken form (`num2words` or a small Gemma pass) per target language. XTTS currently mangles "2025", "km", "Dr.", etc.
2. **Visual Lip-Syncing:** Integrate **Wav2Lip** or **LivePortrait** to re-animate mouths to match the new audio. Largest perceptual leap; also the largest scope (model download, GPU cost, failure modes on non-frontal shots).
3. **Diarization Robustness:** Merge over-split pyannote speakers via cosine similarity on speaker embeddings; re-verify speaker assignments near boundaries.
4. **Batch Processing:** Ability to queue multiple videos in the web dashboard.
5. **Translation Editor:** Interactive UI to manually override LLM translations before synthesis.

## Verification & Testing
1. **Single-Speaker Test:** Verified with Spanish-to-English sample.
2. **Multi-Speaker Test:** Verified with 5-minute technical interview (diarization and triangulation confirmed).
3. **Multimodal Emotion Test:** Verified that Gemma 4 detects sarcasm/tone and XTTS v2 reflects it without speaking the tags.

## Resiliency & State Management
- **Checkpointing:** Pipeline skips already-completed stages (Extraction, Separation, Transcription, Translation, Synthesis) based on existing file artifacts.
- **Granular Resumption:** Segment-level synthesis resumption to handle crashes during long renders.
