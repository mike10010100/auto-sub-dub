import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from pydub import AudioSegment

from src.utils import get_device

logger = logging.getLogger(__name__)


class BaseSynthesizer:
    def __init__(self, output_dir="output/audio_segments", device=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_dir = Path("output/references")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
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
        Score a candidate reference clip for cloning. Returns a dict with
        voiced_ratio, snr_db, dbfs.
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
            mask[seg["start"] : seg["end"]] = True

        voiced = samples[mask]
        unvoiced = samples[~mask]
        voiced_ratio = voiced.size / samples.size

        voiced_rms = float(np.sqrt(np.mean(voiced * voiced) + 1e-12)) if voiced.size else 1e-6
        unvoiced_rms = (
            float(np.sqrt(np.mean(unvoiced * unvoiced) + 1e-12)) if unvoiced.size else 1e-6
        )
        snr_db = 20.0 * np.log10(voiced_rms / max(unvoiced_rms, 1e-6))

        return {"voiced_ratio": voiced_ratio, "snr_db": snr_db, "dbfs": clip.dBFS}

    def extract_speaker_references(
        self,
        vocals_path,
        transcript,
        target_clips=3,
        min_duration=3,
        max_duration=20,
        trim_to=12.0,
        hq_vocals_path=None,
        min_voiced_ratio=0.55,
    ):
        """
        Extract per-speaker reference clips for voice cloning.
        Returns: {speaker: [{"path": str, "emotion": str, "text": str}, ...]}
        """
        source_path = hq_vocals_path or vocals_path
        logger.info(
            f"Extracting multi-reference samples from {source_path} "
            f"(target: {target_clips} clips, min/max seg duration: {min_duration}/{max_duration}s)"
        )
        audio = AudioSegment.from_wav(source_path)
        index_path = self.ref_audio_dir / "references.json"

        # Use raw segments for references since they match the source audio timing
        segments = transcript.get("segments", [])

        if index_path.exists():
            try:
                cached = json.loads(index_path.read_text(encoding="utf-8"))
                if all(Path(c["path"]).exists() for clips in cached.values() for c in clips):
                    logger.info(f"Using cached references from {index_path}")
                    return cached
            except Exception as e:
                logger.warning(f"Could not read ref index {index_path}: {e}")

        references = {}
        unique_speakers = {s["speaker"] for s in segments if s.get("speaker")}

        for speaker in unique_speakers:
            spk_segs = [s for s in segments if s.get("speaker") == speaker]
            candidates = []
            for seg in spk_segs:
                duration = seg["end"] - seg["start"]
                if duration < min_duration:
                    continue
                start_ms = int(seg["start"] * 1000)
                end_ms = int(min(seg["end"], seg["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                # Relaxed thresholds: -45dBFS and 0.40 voiced ratio
                if clip.dBFS <= -45:
                    continue
                metrics = self._score_reference_clip(clip)
                if metrics["voiced_ratio"] < 0.40:
                    continue
                # Emotion might not be in raw segments, so we look it up in translated
                # or just use NEUTRAL for reference extraction.
                emotion = seg.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                candidates.append((score, clip, emotion, seg.get("text", ""), metrics))

            if not candidates and spk_segs:
                # Even more relaxed fallback for rare side-characters
                longest = max(spk_segs, key=lambda s: s["end"] - s["start"])
                start_ms = int(longest["start"] * 1000)
                end_ms = int(min(longest["end"], longest["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                metrics = self._score_reference_clip(clip)
                emotion = longest.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                logger.warning(
                    f"  {speaker}: using low-quality fallback reference (voiced={metrics['voiced_ratio']:.2f})"
                )
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                candidates.append((score, clip, emotion, longest.get("text", ""), metrics))

            candidates.sort(key=lambda x: x[0], reverse=True)
            picked = candidates[:target_clips]

            entries = []
            for i, (score, clip, emotion, text, metrics) in enumerate(picked):
                safe_emotion = emotion.strip("[]") or "NEUTRAL"
                ref_path = self.ref_audio_dir / f"{speaker}_ref_{i}_{safe_emotion}.wav"
                clip.export(ref_path, format="wav")
                entries.append({"path": str(ref_path), "emotion": emotion, "text": text})

            if entries:
                references[speaker] = entries
                logger.info(f"  {speaker}: extracted {len(entries)} reference clips.")

        index_path.write_text(json.dumps(references, indent=2), encoding="utf-8")
        return references

    def adjust_speed(self, audio_path, target_duration):
        """Adjusts the speed of an audio file to match the target duration without changing pitch."""
        with (
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out,
        ):
            temp_in_path = temp_in.name
            temp_out_path = temp_out.name

            audio = AudioSegment.from_file(audio_path)
            current_duration = audio.duration_seconds

            # Ensure the audio is Mono, 44100Hz, 16-bit PCM for maximum compatibility with audiotsm
            audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
            audio.export(temp_in_path, format="wav", codec="pcm_s16le")

            speed_ratio = current_duration / target_duration

            if speed_ratio < 1.0 or speed_ratio > 2.0:
                logger.info(f"Speed ratio {speed_ratio:.2f} clipped to [1.0, 2.0].")
                speed_ratio = max(1.0, min(2.0, speed_ratio))

            if abs(speed_ratio - 1.0) < 0.01:
                logger.info("Speed ratio is ~1.0, skipping WSOLA to prevent artifacts.")
                audio.export(audio_path, format="wav")
                if os.path.exists(temp_in_path):
                    os.unlink(temp_in_path)
                if os.path.exists(temp_out_path):
                    os.unlink(temp_out_path)
                return audio_path

            try:
                # Use a custom WavReader if needed, but standard should work if format is fixed
                with WavReader(temp_in_path) as reader:
                    with WavWriter(temp_out_path, reader.channels, reader.samplerate) as writer:
                        tsm = wsola(reader.channels, speed=speed_ratio)
                        tsm.run(reader, writer)

                final_audio = AudioSegment.from_wav(temp_out_path)
                final_audio.export(audio_path, format="wav")
            except Exception as e:
                logger.error(f"Time-stretching failed ({e}); falling back to simple speed change.")
                # High-quality fallback if WSOLA fails: simple resampling (changes pitch, but is safe)
                # Or just use the original if pitch change is undesirable.
                # Here we stick to original to maintain quality.
                pass

            if os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if os.path.exists(temp_out_path):
                os.unlink(temp_out_path)

            return audio_path


class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, output_dir="output/audio_segments", device=None):
        super().__init__(output_dir, device)

        # Force CPU for XTTS on Mac
        if self.device == "mps":
            logger.info("XTTS v2 is unstable on MPS. Forcing CPU for high-quality synthesis.")
            self.device = "cpu"

        self.model = None

    def _load_model(self):
        if self.model is None:
            from TTS.api import TTS

            logger.info(f"Loading XTTS v2 model on {self.device}...")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def synthesize(
        self, text, speaker_id, speaker_refs, output_filename, language="en", emotion=None
    ):
        self._load_model()
        output_path = self.output_dir / output_filename
        ref_paths = [r["path"] for r in speaker_refs if r["emotion"] == emotion] or [
            r["path"] for r in speaker_refs
        ]

        if not ref_paths:
            return None

        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref_paths,
                language=language,
                file_path=str(output_path),
            )
            return output_path
        except Exception as e:
            logger.error(f"XTTS Synthesis failed: {e}")
            return None


class FishSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        output_dir="output/audio_segments",
        device=None,
        s2_cpp_path=None,
        model_path="models/s2-pro-q4_k_m.gguf",
        tokenizer_path="models/tokenizer.json",
    ):
        super().__init__(output_dir, device)

        # Check standard locations for s2 binary
        candidates = ["./s2.cpp/build/s2", "../s2.cpp/build/s2", "/usr/local/bin/s2", "s2"]

        if s2_cpp_path:
            candidates.insert(0, s2_cpp_path)

        self.s2_cpp_path = None
        for cand in candidates:
            if cand == "s2":  # Check PATH
                if subprocess.run(["which", "s2"], capture_output=True).returncode == 0:
                    self.s2_cpp_path = "s2"
                    break
            elif os.path.exists(cand):
                self.s2_cpp_path = cand
                break

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        if not self.s2_cpp_path:
            logger.error("s2.cpp binary not found in expected locations.")
        else:
            logger.info(f"Using Fish Speech binary: {self.s2_cpp_path}")

    def synthesize(
        self,
        text,
        speaker_id,
        speaker_refs,
        output_filename,
        language="en",
        emotion=None,
        temp=0.7,
        top_p=0.8,
        top_k=20,
    ):
        output_path = self.output_dir / output_filename
        # Pick the best reference for Fish Speech
        ref_entry = next((r for r in speaker_refs if r["emotion"] == emotion), speaker_refs[0])
        ref_wav = ref_entry["path"]
        ref_text = ref_entry.get("text", "")

        # Format the text with emotion tags for Fish Speech
        formatted_text = f"{emotion} {text}" if emotion and emotion.startswith("[") else text

        cmd = [
            self.s2_cpp_path,
            "-m",
            self.model_path,
            "-t",
            self.tokenizer_path,
            "-text",
            formatted_text,
            "-pa",
            ref_wav,
            "-pt", ref_text,
            "-o", str(output_path),
            "-temp", str(temp),
            "-top-p", str(top_p),
            "-top-k", str(top_k),
            "--trim-silence",
            ]


        if self.device == "cuda":
            cmd.extend(["-c", "0"])
        elif self.device == "mps":
            cmd.append("-M")

        logger.info(f"Synthesizing with Fish Speech: {formatted_text}")
        try:
            # We must use CUDA if possible. s2.cpp uses Vulkan/CUDA internally
            # if compiled with it.
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Fish Speech Synthesis failed: {e.stderr.decode()}")
            return None


def Synthesizer(engine="xtts", **kwargs):
    if engine == "fish":
        return FishSynthesizer(**kwargs)
    return XTTSSynthesizer(**kwargs)
  ]

        if self.device == "cuda":
            cmd.extend(["-c", "0"])
        elif self.device == "mps":
            cmd.append("-M")

        logger.info(f"Synthesizing with Fish Speech: {formatted_text}")
        try:
            # We must use CUDA if possible. s2.cpp uses Vulkan/CUDA internally
            # if compiled with it.
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Fish Speech Synthesis failed: {e.stderr.decode()}")
            return None


def Synthesizer(engine="xtts", **kwargs):
    if engine == "fish":
        return FishSynthesizer(**kwargs)
    return XTTSSynthesizer(**kwargs)
