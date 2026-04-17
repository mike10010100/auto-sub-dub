import os
import argparse
import json
from pathlib import Path
from pydub import AudioSegment
import subprocess

from src.audio_processor import AudioProcessor
from src.transcriber import Transcriber
from src.translator import Translator
from src.synthesizer import Synthesizer

def main(video_path, target_lang="Spanish", hf_token=None):
    # 1. Initialize components
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Priority: CLI argument > .env environment variable
    hf_token = hf_token or os.getenv("HF_TOKEN")
    
    audio_proc = AudioProcessor(output_dir=output_dir)
    transcriber = Transcriber(hf_token=hf_token)
    translator = Translator()
    synthesizer = Synthesizer(output_dir=output_dir / "audio_segments")
    
    # 2. Process Audio
    orig_audio = audio_proc.extract_audio(video_path)
    vocals, background = audio_proc.separate_vocals(orig_audio)
    
    # 3. Transcribe & Diarize
    transcript = transcriber.transcribe(vocals)
    transcriber.save_transcript(transcript, output_dir / "transcript.json")
    
    # 4. Translate
    translated_segments = translator.translate_segments(transcript["segments"], target_lang=target_lang)
    transcript["translated_segments"] = translated_segments
    transcriber.save_transcript(transcript, output_dir / "transcript_translated.json")
    
    # 5. Extract Speaker References
    references = synthesizer.extract_speaker_references(vocals, transcript)
    
    # 6. Synthesize & Place Audio
    # Mapping for TTS (Coqui expects 2-letter codes)
    lang_map = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Polish": "pl",
        "Turkish": "tr",
        "Russian": "ru",
        "Dutch": "nl",
        "Czech": "cs",
        "Arabic": "ar",
        "Chinese": "zh-cn",
        "Japanese": "ja",
        "Korean": "ko",
        "Hungarian": "hu",
        "Hindi": "hi"
    }
    tts_lang = lang_map.get(target_lang, "en")
    
    # Initialize blank audio matching original length
    original_audio = AudioSegment.from_wav(orig_audio)
    dubbed_audio_track = AudioSegment.silent(duration=len(original_audio))
    
    for i, segment in enumerate(translated_segments):
        speaker = segment.get("speaker")
        text = segment.get("text")
        start_time = segment.get("start")
        end_time = segment.get("end")
        
        if not speaker or not text or speaker not in references:
            continue
            
        # Synthesize clip
        clip_name = f"segment_{i}_{speaker}.wav"
        clip_path = synthesizer.synthesize(text, speaker, references[speaker], clip_name, language=tts_lang)
        
        if not clip_path or not os.path.exists(clip_path):
            print(f"Skipping segment {i} (Synthesized file not found)")
            continue
            
        # Adjust speed to fit segment duration
        target_duration = end_time - start_time
        synthesizer.adjust_speed(clip_path, target_duration)
        
        # Load and overlay
        segment_audio = AudioSegment.from_wav(clip_path)
        start_ms = int(start_time * 1000)
        dubbed_audio_track = dubbed_audio_track.overlay(segment_audio, position=start_ms)
        
    # 7. Mix with background and Remux
    print("Mixing final audio track...")
    background_audio = AudioSegment.from_wav(background)
    final_mixed_audio = dubbed_audio_track.overlay(background_audio)
    
    final_audio_path = output_dir / "final_audio.wav"
    final_mixed_audio.export(final_audio_path, format="wav")
    
    # Remux back to video
    video_output_path = output_dir / f"dubbed_{Path(video_path).name}"
    print(f"Remuxing final video to {video_output_path}...")
    
    # FFmpeg command to replace audio stream
    subprocess.run([
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(final_audio_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(video_output_path)
    ], check=True)
    
    print("Auto-Dubbing Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Dub Pipeline")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--lang", default="Spanish", help="Target language for translation")
    parser.add_argument("--hf_token", help="Hugging Face token for diarization")
    
    args = parser.parse_args()
    main(args.video, target_lang=args.lang, hf_token=args.hf_token)
