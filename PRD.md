# Auto-Dub PRD & Architecture Plan

## Objective
Build a fully local, open-source pipeline that takes a video file, isolates vocals, transcribes/diarizes the audio, translates the transcript using a local LLM (Gemma 4 via Ollama), generates expressive dubbed audio using voice cloning, and remuxes the audio back into the video.

## Key Files & Context
- **Repository:** `/home/mike10010100/git/auto-sub-dub`
- **LLM Endpoint:** `http://192.168.86.172:11434` (Ollama running Gemma 4 26B)
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

### Phase 5: Productionization (In Progress)
1. **Sidechain Compression (Vocal Ducking):** Automatically lower background music volume when speech is active.
2. **Web-Based Interface:** Build a **Streamlit** dashboard for drag-and-drop video processing and manual translation review.
3. **Visual Lip-Syncing:** Integrate **Wav2Lip** or **LivePortrait** to re-animate mouths to match the new English audio.
4. **Enterprise Logging:** Transition to structured logging and robust error handling for multi-hour video processing.

## Verification & Testing
1. **Single-Speaker Test:** Verified with Spanish-to-English sample.
2. **Multi-Speaker Test:** Verified with 5-minute technical interview (diarization and triangulation confirmed).
3. **Multimodal Emotion Test:** Verified that Gemma 4 detects sarcasm/tone and XTTS v2 reflects it without speaking the tags.

## Resiliency & State Management
- **Checkpointing:** Pipeline skips already-completed stages (Extraction, Separation, Transcription, Translation, Synthesis) based on existing file artifacts.
- **Granular Resumption:** Segment-level synthesis resumption to handle crashes during long renders.
