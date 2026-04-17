import whisperx
import json
from pathlib import Path
import logging
from src.utils import get_device, get_compute_type

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self, device=None, compute_type=None, hf_token=None):
        self.device = device or get_device()
        self.compute_type = compute_type or get_compute_type(self.device)
        self.hf_token = hf_token
        
        # WhisperX specific fix for MPS/CPU (it can sometimes fail with float16)
        if self.device != "cuda" and self.compute_type == "float16":
            logger.warning(f"Forcing compute_type='float32' for {self.device}")
            self.compute_type = "float32"
            
        logger.info(f"Initialized Transcriber on {self.device} (compute_type={self.compute_type})")

    def transcribe(self, audio_path, batch_size=16):
        """Transcribes the audio and performs speaker diarization."""
        logger.info(f"Transcribing {audio_path}...")
        
        # 1. Transcribe with original whisper
        model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type)
        audio = whisperx.load_audio(str(audio_path))
        result = model.transcribe(audio, batch_size=batch_size)
        
        # 2. Align whisper output
        logger.info("Aligning transcription...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        
        # 3. Assign speaker labels
        logger.info("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)
        
        # 4. Merge results
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        return result

    def save_transcript(self, transcript, output_path):
        """Saves the diarized transcript to a JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"Transcript saved to {output_path}")

if __name__ == "__main__":
    # Test stub
    import sys
    import os
    if len(sys.argv) > 1:
        # Check for HF token in env
        hf_token = os.getenv("HF_TOKEN")
        transcriber = Transcriber(hf_token=hf_token)
        transcript = transcriber.transcribe(sys.argv[1])
        transcriber.save_transcript(transcript, "transcript.json")
