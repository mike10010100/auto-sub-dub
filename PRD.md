# Auto-Dub PRD & Architecture Plan

## Objective
Build a fully local, open-source pipeline that takes a video file, isolates vocals, transcribes/diarizes the audio, translates the transcript using a local LLM (Gemma 4 via Ollama), generates expressive dubbed audio using voice cloning, and remuxes the audio back into the video.

## Key Files & Context
- **Repository:** `/home/mike10010100/git/auto-sub-dub`
- **LLM Endpoint:** `http://192.168.86.172:11434` (Ollama running Gemma 4)
- **Hardware Profile:** NVIDIA RTX 3080 (10GB VRAM) for local processing, RTX 3500 Ada for Ollama.

## Architecture & Technology Stack
1. **Audio Extraction & Mixing:** `FFmpeg`
2. **Vocal Isolation:** `Demucs` - to separate vocals from background noise/music.
3. **Transcription & Diarization:** `WhisperX` - High-speed transcription with speaker diarization via `pyannote.audio`.
4. **Translation Engine:** `Gemma 4` (via local Ollama API) - Duration-aware prompting to match original segment lengths.
5. **Expressive Voice Cloning (TTS):** `Coqui XTTS v2` - Using **Multi-Reference Triangulation** (3+ clips per speaker) for high fidelity.
6. **Audio Stitching & Time-Stretching:** `pydub` and `audiotsm` (WSOLA) - For precise timestamp placement and speed adjustment.

## Implementation Phases

### Phase 1: Setup and Basic Extraction (Completed)
1. Initialize Python project structure and `requirements.txt`.
2. Implement `AudioProcessor` for audio extraction and `Demucs` vocal isolation.

### Phase 2: Transcription and Translation (Completed)
1. Implement `Transcriber` using `WhisperX` for diarized transcripts.
2. Implement `Translator` with duration-aware prompting for `Gemma 4`.

### Phase 3: Voice Cloning & Audio Synthesis (Completed)
1. Implement `Synthesizer` with multi-reference triangulation (XTTS v2).
2. Apply WSOLA time-stretching to match original spoken durations.

### Phase 4: Assembly and Final Render (Completed)
1. Per-video project directories for output isolation.
2. Automatic overlay mixing and `FFmpeg` remuxing.

### Phase 5: Productionization (Planned)
1. **Sidechain Compression (Vocal Ducking):** Automatically lower background music volume when speech is active.
2. **Dynamic Emotional Cues:** Pass context-based emotion tags (`[ANGRY]`, `[EXCITED]`) from Gemma 4 to XTTS v2.
3. **Web-Based Interface:** Build a **Streamlit** dashboard for drag-and-drop video processing and manual translation review.
4. **Visual Lip-Syncing:** Integrate **Wav2Lip** or **LivePortrait** to re-animate mouths to match the new English audio.
5. **Enterprise Logging:** Transition to structured logging and robust error handling for multi-hour video processing.

## Verification & Testing
1. **Single-Speaker Test:** Verified with Spanish-to-English sample.
2. **Multi-Speaker Test:** Verified with 5-minute technical interview (diarization and triangulation confirmed).
3. **Performance Metrics:** Ensure synthesis stays near real-time on local GPU.

## Resiliency & State Management
- **Checkpointing:** Pipeline skips already-completed stages (Extraction, Separation, Transcription, Translation, Synthesis) based on existing file artifacts.
- **Granular Resumption:** Segment-level synthesis resumption to handle crashes during long renders.
