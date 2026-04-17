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

    def extract_speaker_references(self, vocals_path, transcript, target_clips=3, min_duration=5, max_duration=12):
        """Extracts multiple discrete high-quality audio clips per speaker for triangulation."""
        logger.info(f"Extracting multi-reference samples from {vocals_path} (Target: {target_clips} clips)...")
        audio = AudioSegment.from_wav(vocals_path)
        
        # Identify speakers and check for existing references
        unique_speakers = set()
        for segment in transcript.get("segments", []):
            if segment.get("speaker"):
                unique_speakers.add(segment["speaker"])
        
        references = {}
        for speaker in unique_speakers:
            speaker_ref_paths = []
            # Check if references already exist for this speaker
            for i in range(target_clips):
                ref_path = self.ref_audio_dir / f"{speaker}_ref_{i}.wav"
                if ref_path.exists():
                    speaker_ref_paths.append(str(ref_path))
            
            if speaker_ref_paths:
                logger.info(f"Using {len(speaker_ref_paths)} existing triangulation references for {speaker}")
                references[speaker] = speaker_ref_paths
                continue

            # If not existing, extract them
            clips = []
            for segment in transcript.get("segments", []):
                if segment.get("speaker") == speaker:
                    duration = segment["end"] - segment["start"]
                    if min_duration <= duration <= max_duration:
                        start_ms = int(segment["start"] * 1000)
                        end_ms = int(segment["end"] * 1000)
                        clip = audio[start_ms:end_ms]
                        if clip.dBFS > -40: 
                            clips.append(clip)
            
            # Sort by quality (loudness/energy) and take the top N
            clips.sort(key=lambda x: x.dBFS, reverse=True)
            top_clips = clips[:target_clips]
            
            extracted_paths = []
            for i, clip in enumerate(top_clips):
                ref_path = self.ref_audio_dir / f"{speaker}_ref_{i}.wav"
                clip.export(ref_path, format="wav")
                extracted_paths.append(str(ref_path))
            
            if extracted_paths:
                references[speaker] = extracted_paths
                logger.info(f"Saved {len(extracted_paths)} NEW triangulation references for {speaker}")
        
        return references

    def synthesize(self, text, speaker_id, ref_audio_paths, output_filename, language="en", emotion=None):
        """Synthesizes text into audio using multiple reference audio clips for better triangulation."""
        self._load_model()
        
        output_path = self.output_dir / output_filename
        
        logger.info(f"Synthesizing for {speaker_id} in {language} (Emotion: {emotion})")
        
        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref_audio_paths,
                language=language,
                file_path=str(output_path),
                emotion=emotion
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
            
            # Avoid extreme stretching
            if speed_ratio < 0.5 or speed_ratio > 2.0:
                logger.warning(f"Speed ratio {speed_ratio:.2f} is extreme. Clipping to [0.5, 2.0].")
                speed_ratio = max(0.5, min(2.0, speed_ratio))
            
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
