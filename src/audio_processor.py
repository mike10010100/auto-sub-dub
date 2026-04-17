import os
import subprocess
import ffmpeg
import demucs.separate
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path):
        """Extracts the audio track from a video file."""
        logger.info(f"Extracting audio from {video_path}...")
        audio_path = self.temp_dir / "original_audio.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(str(audio_path), acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
            raise

    def separate_vocals(self, audio_path):
        """Separates vocals from background audio using Demucs."""
        logger.info(f"Separating vocals using Demucs for {audio_path}...")
        # Demucs CLI is often easier to use directly from Python
        # We use the 'htdemucs' model by default
        try:
            subprocess.run([
                "demucs",
                "--two-stems", "vocals",
                "-o", str(self.temp_dir),
                str(audio_path)
            ], check=True)
            
            # Demucs creates a folder structure: output/temp/htdemucs/original_audio/vocals.wav
            # and output/temp/htdemucs/original_audio/no_vocals.wav
            base_name = Path(audio_path).stem
            model_name = "htdemucs"
            
            vocals_path = self.temp_dir / model_name / base_name / "vocals.wav"
            background_path = self.temp_dir / model_name / base_name / "no_vocals.wav"
            
            return vocals_path, background_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs separation failed: {e}")
            raise

    def duck_audio(self, vocals, background, duck_db=-15, attack_ms=50, release_ms=200):
        """
        Performs vocal ducking: lowers the background volume when vocals are active.
        'vocals' and 'background' are pydub AudioSegment objects.
        """
        from pydub import AudioSegment
        
        logger.info(f"Applying vocal ducking (reduction: {duck_db}dB)...")
        
        # Ensure they are the same length
        max_len = max(len(vocals), len(background))
        vocals = vocals + AudioSegment.silent(duration=max_len - len(vocals))
        background = background + AudioSegment.silent(duration=max_len - len(background))
        
        # Chunk size for analysis (e.g., 20ms)
        chunk_size = 20
        ducked_background = AudioSegment.empty()
        
        # Simple implementation: check RMS of vocal chunks
        # A more sophisticated version would use attack/release envelopes
        # But this works well for basic ducking.
        for i in range(0, len(vocals), chunk_size):
            vocal_chunk = vocals[i:i+chunk_size]
            bg_chunk = background[i:i+chunk_size]
            
            # If vocal chunk is louder than silence threshold (e.g., -40dB)
            if vocal_chunk.dBFS > -40:
                # Apply ducking
                ducked_background += bg_chunk + duck_db
            else:
                ducked_background += bg_chunk
                
        return vocals.overlay(ducked_background)

if __name__ == "__main__":
    # Test stub
    import sys
    if len(sys.argv) > 1:
        proc = AudioProcessor()
        orig_audio = proc.extract_audio(sys.argv[1])
        v, b = proc.separate_vocals(orig_audio)
        print(f"Vocals: {v}")
        print(f"Background: {b}")
