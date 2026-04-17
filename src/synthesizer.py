import os
from pathlib import Path
from pydub import AudioSegment
from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
import subprocess
import tempfile

class Synthesizer:
    def __init__(self, output_dir="output/audio_segments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_dir = Path("output/references")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)

    def extract_speaker_references(self, vocals_path, transcript):
        """Extracts a short audio sample for each unique speaker to use as a cloning reference."""
        print(f"Extracting speaker references from {vocals_path}...")
        audio = AudioSegment.from_wav(vocals_path)
        
        speakers = set()
        for segment in transcript.get("segments", []):
            speaker = segment.get("speaker")
            if speaker and speaker not in speakers:
                # Find a segment of decent length (e.g., 5-10 seconds)
                duration = segment["end"] - segment["start"]
                if 5 <= duration <= 15:
                    start_ms = int(segment["start"] * 1000)
                    end_ms = int(segment["end"] * 1000)
                    ref_clip = audio[start_ms:end_ms]
                    ref_path = self.ref_audio_dir / f"{speaker}_ref.wav"
                    ref_clip.export(ref_path, format="wav")
                    speakers.add(speaker)
                    print(f"Saved reference for {speaker} to {ref_path}")
        
        return {s: self.ref_audio_dir / f"{s}_ref.wav" for s in speakers}

    def synthesize(self, text, speaker_id, ref_audio_path, output_filename):
        """Synthesizes text into audio using the specified speaker's reference audio."""
        # This is a placeholder for the actual Fish Speech or XTTS call.
        # For now, we'll use a subprocess call to a hypothetical Fish Speech CLI or API.
        
        output_path = self.output_dir / output_filename
        
        # Example call to XTTS (as it's more standard to implement in a script)
        # In a real scenario, this would be replaced by a call to the Fish Speech API
        print(f"Synthesizing '{text[:30]}...' for {speaker_id}")
        
        # Hypothetical CLI usage:
        # subprocess.run(["fish-speech", "--text", text, "--ref", str(ref_audio_path), "--out", str(output_path)])
        
        # For the prototype, we'll just log the action. 
        # In a real implementation, you'd use the specific API of the chosen model.
        return output_path

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
