import os
from pathlib import Path
from pydub import AudioSegment
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
import tempfile
import torch
from TTS.api import TTS
import logging
from src.utils import get_device

logger = logging.getLogger(__name__)

class Synthesizer:
    def __init__(self, output_dir="output/audio_segments", device=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_dir = Path("output/references")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine device
        detected_device = device or get_device()
        
        # Force CPU for XTTS on Mac (MPS is too unstable for this specific model)
        if detected_device == "mps":
            logger.info("XTTS v2 is unstable on MPS. Forcing CPU for high-quality synthesis.")
            self.device = "cpu"
        else:
            self.device = detected_device
        
        logger.info(f"Initialized Synthesizer on {self.device}")

        self.model = None
        self._vad = None
        self._vad_utils = None

    def _load_vad(self):
        """Lazy-load Silero VAD. Returns (model, utils) or (None, None) on failure."""
        if self._vad is not None:
            return self._vad, self._vad_utils
        try:
            import torch as _torch
            vad, utils = _torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad = vad
            self._vad_utils = utils
            return vad, utils
        except Exception as e:
            logger.warning(f"Silero VAD unavailable ({e}); falling back to dBFS ranking.")
            return None, None

    def _score_reference_clip(self, clip):
        """
        Score a candidate reference clip for XTTS cloning. Returns a dict with
        voiced_ratio, snr_db, dbfs. VAD gates out clips that are mostly music
        or silence after Demucs; SNR favors clips where the voice sits well
        above residual background/noise.
        """
        import numpy as np
        vad, utils = self._load_vad()
        if vad is None:
            return {"voiced_ratio": 1.0, "snr_db": 0.0, "dbfs": clip.dBFS}

        mono_16k = clip.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        samples = np.frombuffer(mono_16k.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < 16000 * 0.5:
            return {"voiced_ratio": 0.0, "snr_db": -60.0, "dbfs": clip.dBFS}

        import torch as _torch
        get_speech_timestamps = utils[0]
        tensor = _torch.from_numpy(samples)
        try:
            ts = get_speech_timestamps(tensor, vad, sampling_rate=16000)
        except Exception as e:
            logger.warning(f"VAD failed on clip: {e}")
            return {"voiced_ratio": 1.0, "snr_db": 0.0, "dbfs": clip.dBFS}

        if not ts:
            return {"voiced_ratio": 0.0, "snr_db": -60.0, "dbfs": clip.dBFS}

        mask = np.zeros(samples.shape[0], dtype=bool)
        for seg in ts:
            mask[seg["start"]:seg["end"]] = True

        voiced = samples[mask]
        unvoiced = samples[~mask]
        voiced_ratio = voiced.size / samples.size

        voiced_rms = float(np.sqrt(np.mean(voiced * voiced) + 1e-12)) if voiced.size else 1e-6
        unvoiced_rms = float(np.sqrt(np.mean(unvoiced * unvoiced) + 1e-12)) if unvoiced.size else 1e-6
        snr_db = 20.0 * np.log10(voiced_rms / max(unvoiced_rms, 1e-6))

        return {"voiced_ratio": voiced_ratio, "snr_db": snr_db, "dbfs": clip.dBFS}

    def _load_model(self):
        """Lazy load the TTS model only when needed."""
        if self.model is None:
            logger.info(f"Loading XTTS v2 model on {self.device}...")
            try:
                # We use XTTS v2 for expressive zero-shot voice cloning
                self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            except Exception as e:
                if self.device == "mps":
                    logger.warning(f"Failed to load XTTS on MPS: {e}. Falling back to CPU.")
                    self.device = "cpu"
                    self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
                else:
                    raise

    def extract_speaker_references(self, vocals_path, transcript, target_clips=3,
                                    min_duration=3, max_duration=20,
                                    trim_to=12.0,
                                    hq_vocals_path=None,
                                    min_voiced_ratio=0.55):
        """
        Extract per-speaker reference clips for XTTS voice cloning. Each clip
        carries the emotion tag of the source segment (from the translated
        transcript) so synthesis can pick a clip that matches the target
        emotion — XTTS's own `emotion=` kwarg doesn't meaningfully condition
        output, but the reference-clip affect DOES carry through.

        Returns: {speaker: [{"path": str, "emotion": str}, ...]}
        """
        import json as _json

        source_path = hq_vocals_path or vocals_path
        logger.info(
            f"Extracting multi-reference samples from {source_path} "
            f"(target: {target_clips} clips, min/max seg duration: {min_duration}/{max_duration}s)"
        )
        audio = AudioSegment.from_wav(source_path)
        index_path = self.ref_audio_dir / "references.json"

        # Prefer the translated segments (which carry emotion); fall back to raw.
        segments = transcript.get("translated_segments") or transcript.get("segments", [])

        # Resume from sidecar if present.
        if index_path.exists():
            try:
                cached = _json.loads(index_path.read_text(encoding="utf-8"))
                if all(Path(c["path"]).exists() for clips in cached.values() for c in clips):
                    logger.info(f"Using cached references from {index_path}")
                    return cached
            except Exception as e:
                logger.warning(f"Could not read ref index {index_path}: {e}")

        references = {}
        unique_speakers = {s["speaker"] for s in segments if s.get("speaker")}

        for speaker in unique_speakers:
            spk_segs = [s for s in segments if s.get("speaker") == speaker]
            candidates = []  # list of (score_tuple, clip, emotion, metrics)
            for seg in spk_segs:
                duration = seg["end"] - seg["start"]
                if duration < min_duration:
                    continue
                start_ms = int(seg["start"] * 1000)
                # Trim long diarization blobs to the XTTS sweet spot rather
                # than rejecting them outright — otherwise side characters
                # whose only segments are 15-30s monologues get zero refs.
                end_ms = int(min(seg["end"], seg["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                if clip.dBFS <= -40:
                    continue
                metrics = self._score_reference_clip(clip)
                if metrics["voiced_ratio"] < min_voiced_ratio:
                    continue
                emotion = seg.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                candidates.append((score, clip, emotion, metrics))

            # Fallback: no clip passed the quality filters — don't let this
            # speaker go silent in the dub. Take the longest segment,
            # trim to the XTTS sweet spot, skip VAD gating, and accept it.
            if not candidates and spk_segs:
                longest = max(spk_segs, key=lambda s: s["end"] - s["start"])
                start_ms = int(longest["start"] * 1000)
                end_ms = int(min(longest["end"], longest["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                metrics = self._score_reference_clip(clip)
                emotion = longest.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                logger.warning(
                    f"  {speaker}: no clips passed filters; falling back to "
                    f"longest segment ({(longest['end']-longest['start']):.1f}s "
                    f"trimmed to {trim_to}s, voiced={metrics['voiced_ratio']:.2f})"
                )
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                candidates.append((score, clip, emotion, metrics))

            candidates.sort(key=lambda x: x[0], reverse=True)
            picked = candidates[:target_clips]

            entries = []
            for i, (score, clip, emotion, metrics) in enumerate(picked):
                logger.info(
                    f"  {speaker} ref {i}: voiced={metrics['voiced_ratio']:.2f} "
                    f"snr={metrics['snr_db']:.1f}dB dBFS={metrics['dbfs']:.1f} emotion={emotion}"
                )
                # Encode emotion in filename so references are self-describing.
                safe_emotion = emotion.strip("[]") or "NEUTRAL"
                ref_path = self.ref_audio_dir / f"{speaker}_ref_{i}_{safe_emotion}.wav"
                clip.export(ref_path, format="wav")
                entries.append({"path": str(ref_path), "emotion": emotion})

            if entries:
                references[speaker] = entries
                emotions_seen = sorted({e["emotion"] for e in entries})
                logger.info(
                    f"Saved {len(entries)} references for {speaker} "
                    f"(emotions: {', '.join(emotions_seen)})"
                )

        index_path.write_text(_json.dumps(references, indent=2), encoding="utf-8")
        return references

    @staticmethod
    def _select_references(speaker_refs, emotion):
        """Pick the clips whose emotion matches; fall back to all clips."""
        if not speaker_refs:
            return []
        matches = [r["path"] for r in speaker_refs if r["emotion"] == emotion]
        if matches:
            return matches
        return [r["path"] for r in speaker_refs]

    def synthesize(self, text, speaker_id, speaker_refs, output_filename, language="en", emotion=None):
        """
        Synthesize text via XTTS v2. `speaker_refs` is the list of
        {"path":..., "emotion":...} dicts returned by extract_speaker_references.
        We pick the subset whose source-emotion matches `emotion`, so the
        cloned voice naturally inherits that delivery — XTTS's own emotion
        kwarg is effectively a no-op.
        """
        self._load_model()
        output_path = self.output_dir / output_filename
        ref_paths = self._select_references(speaker_refs, emotion)
        if not ref_paths:
            logger.error(f"No references available for {speaker_id}")
            return None

        logger.info(
            f"Synthesizing for {speaker_id} in {language} "
            f"(Emotion: {emotion}, refs: {len(ref_paths)})"
        )
        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref_paths,
                language=language,
                file_path=str(output_path),
            )
            return output_path
        except Exception as e:
            logger.error(f"TTS Synthesis failed for {speaker_id}: {e}")
            return None

    def adjust_speed(self, audio_path, target_duration):
        """Adjusts the speed of an audio file to match the target duration without changing pitch."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out:
            
            # Save the path
            temp_in_path = temp_in.name
            temp_out_path = temp_out.name
            
            # Load and export to temporary file to ensure correct format for audiotsm
            audio = AudioSegment.from_file(audio_path)
            current_duration = audio.duration_seconds
            audio.export(temp_in_path, format="wav")
            
            speed_ratio = current_duration / target_duration

            # Floor at 1.0 — never slow the dub below XTTS's natural pace
            # (sounds sluggish and unnatural), even if the target window is
            # longer. Cap at 2.0 to avoid chipmunk speech when the window
            # is very tight.
            if speed_ratio < 1.0 or speed_ratio > 2.0:
                logger.info(f"Speed ratio {speed_ratio:.2f} clipped to [1.0, 2.0].")
                speed_ratio = max(1.0, min(2.0, speed_ratio))
            
            with WavReader(temp_in_path) as reader:
                with WavWriter(temp_out_path, reader.channels, reader.samplerate) as writer:
                    tsm = wsola(reader.channels, speed=speed_ratio)
                    tsm.run(reader, writer)
            
            # Load adjusted audio and export back to original path
            final_audio = AudioSegment.from_wav(temp_out_path)
            final_audio.export(audio_path, format="wav")
            
            # Clean up
            os.unlink(temp_in_path)
            os.unlink(temp_out_path)
            
            return audio_path

if __name__ == "__main__":
    # Test stub
    logging.basicConfig(level=logging.INFO)
    synth = Synthesizer()
    logger.info("Synthesizer module loaded.")
