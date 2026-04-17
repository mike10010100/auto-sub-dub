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
    video_name = Path(video_path).stem
    project_dir = Path("output") / video_name
    project_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = project_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    audio_segments_dir = project_dir / "audio_segments"
    audio_segments_dir.mkdir(parents=True, exist_ok=True)

    # Priority: CLI argument > .env environment variable
    hf_token = hf_token or os.getenv("HF_TOKEN")
    
    audio_proc = AudioProcessor(output_dir=project_dir)
    transcriber = Transcriber(hf_token=hf_token)
    translator = Translator()
    synthesizer = Synthesizer(output_dir=audio_segments_dir)
    
    # 2. Process Audio
    orig_audio = temp_dir / "original_audio.wav"
    if not orig_audio.exists():
        orig_audio = audio_proc.extract_audio(video_path)
        # The extract_audio method might return a different path if it uses its own logic, 
        # let's ensure we use the one we want.
        # Actually, audio_proc.extract_audio returns self.temp_dir / "original_audio.wav"
    else:
        print(f"Skipping audio extraction, using existing: {orig_audio}")

    # Demucs output paths - updated to be relative to project_dir
    vocals = temp_dir / "htdemucs" / "original_audio" / "vocals.wav"
    background = temp_dir / "htdemucs" / "original_audio" / "no_vocals.wav"
    
    if not vocals.exists() or not background.exists():
        vocals, background = audio_proc.separate_vocals(orig_audio)
    else:
        print(f"Skipping vocal separation, using existing: {vocals}")
    
    # 3. Transcribe & Diarize
    transcript_path = project_dir / "transcript.json"
    if not transcript_path.exists():
        transcript = transcriber.transcribe(vocals)
        transcriber.save_transcript(transcript, transcript_path)
    else:
        print(f"Skipping transcription, using existing: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
    
    # 4. Translate
    translated_transcript_path = project_dir / "transcript_translated.json"
    if not translated_transcript_path.exists():
        # Use Hybrid Multimodal Translation (Audio + Text)
        translated_segments = translator.translate_segments_multimodal(
            transcript["segments"], 
            vocals_path=vocals, 
            target_lang=target_lang
        )
        transcript["translated_segments"] = translated_segments
        transcriber.save_transcript(transcript, translated_transcript_path)
    else:
        print(f"Skipping translation, using existing: {translated_transcript_path}")
        with open(translated_transcript_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        translated_segments = transcript["translated_segments"]
    
    # 5. Extract Speaker References
    # We update the synthesizer to use the project-specific reference dir
    synthesizer.ref_audio_dir = project_dir / "references"
    synthesizer.ref_audio_dir.mkdir(parents=True, exist_ok=True)
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
        clip_path = audio_segments_dir / clip_name
        
        # Pass [EMOTION] tag to synthesizer if available
        emotion_tag = segment.get("emotion", "[NEUTRAL]")
        full_text = f"{emotion_tag} {text}"
        
        if not clip_path.exists():
            clip_path = synthesizer.synthesize(full_text, speaker, references[speaker], clip_name, language=tts_lang)
        else:
            print(f"Skipping synthesis for segment {i}, using existing: {clip_path}")
        
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
    video_output_path = project_dir / f"dubbed_{Path(video_path).name}"

    print("Mixing final audio track...")
    background_audio = AudioSegment.from_wav(background)
    final_mixed_audio = dubbed_audio_track.overlay(background_audio)
    
    final_audio_path = project_dir / "final_audio.wav"
    final_mixed_audio.export(final_audio_path, format="wav")
    
    # Remux back to video
    print(f"Remuxing final video to {video_output_path}...")
    
    # FFmpeg command to replace audio stream
    subprocess.run([
        "ffmpeg",
        "-y", # Overwrite if exists
        "-i", str(video_path),
        "-i", str(final_audio_path),
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(video_output_path)
    ], check=True)
    
    print(f"Auto-Dubbing Complete! Output: {video_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Dub Pipeline")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--lang", default="Spanish", help="Target language for translation")
    parser.add_argument("--hf_token", help="Hugging Face token for diarization")
    
    args = parser.parse_args()
    main(args.video, target_lang=args.lang, hf_token=args.hf_token)
