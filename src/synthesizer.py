import os
from pathlib import Path
from pydub import AudioSegment
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
import tempfile
import torch
from TTS.api import TTS

class Synthesizer:
    def __init__(self, output_dir="output/audio_segments", device="cuda"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_dir = Path("output/references")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None

    def _load_model(self):
        """Lazy load the TTS model only when needed."""
        if self.model is None:
            print(f"Loading XTTS v2 model on {self.device}...")
            # We use XTTS v2 for expressive zero-shot voice cloning
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def extract_speaker_references(self, vocals_path, transcript):
        """Extracts a short audio sample for each unique speaker to use as a cloning reference."""
        print(f"Extracting speaker references from {vocals_path}...")
        audio = AudioSegment.from_wav(vocals_path)
        
        # Merge segments by speaker to get longer reference clips
        speaker_clips = {}
        for segment in transcript.get("segments", []):
            speaker = segment.get("speaker")
            if not speaker:
                continue
            
            if speaker not in speaker_clips:
                speaker_clips[speaker] = []
            
            duration = segment["end"] - segment["start"]
            # We want about 10 seconds of clear speech for a good reference
            if 2 <= duration <= 30: # Use any decent segment
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                speaker_clips[speaker].append(audio[start_ms:end_ms])
        
        references = {}
        for speaker, clips in speaker_clips.items():
            ref_path = self.ref_audio_dir / f"{speaker}_ref.wav"
            if ref_path.exists():
                print(f"Using existing reference for {speaker}: {ref_path}")
                references[speaker] = ref_path
                continue

            if not clips:
                continue
            # Combine the first few clips to reach ~10 seconds
            combined = clips[0]
            for clip in clips[1:]:
                if combined.duration_seconds >= 10:
                    break
                combined += clip
            
            combined.export(ref_path, format="wav")
            references[speaker] = ref_path
            print(f"Saved reference for {speaker} ({combined.duration_seconds:.1f}s) to {ref_path}")
        
        return references

    def synthesize(self, text, speaker_id, ref_audio_path, output_filename, language="en"):
        """Synthesizes text into audio using the specified speaker's reference audio."""
        self._load_model()
        
        output_path = self.output_dir / output_filename
        print(f"Synthesizing '{text[:30]}...' for {speaker_id} in {language}")
        
        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=str(ref_audio_path),
                language=language,
                file_path=str(output_path)
            )
            return output_path
        except Exception as e:
            print(f"TTS Synthesis failed for {speaker_id}: {e}")
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
                print(f"Warning: Speed ratio {speed_ratio:.2f} is extreme. Clipping to [0.5, 2.0].")
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
    synth = Synthesizer()
    print("Synthesizer module loaded.")
