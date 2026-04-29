import argparse
import json
import logging
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from pydub import AudioSegment

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("auto_dub.log")],
)
logger = logging.getLogger(__name__)

# Silence SpeechBrain noise
logging.getLogger("speechbrain").setLevel(logging.ERROR)

from src.audio_processor import AudioProcessor, parse_srt
from src.synthesizer import Synthesizer
from src.timing import annotate_effective_windows  # noqa: E402,F401
from src.transcriber import Transcriber
from src.translator import Translator
from src.utils import get_device


def main(
    video_path,
    target_lang="Spanish",
    hf_token=None,
    device=None,
    ollama_url=None,
    ollama_model=None,
    ollama_audio_model=None,
    engine="xtts",
):  # pragma: no cover
    # 1. Initialize components
    video_name = Path(video_path).stem
    project_dir = Path("output") / video_name
    project_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = project_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    audio_segments_dir = project_dir / "audio_segments"
    audio_segments_dir.mkdir(parents=True, exist_ok=True)

    # Priority: argument > helper > auto-detect
    device = device or get_device()
    logger.info(f"Using device: {device}")

    # Priority: argument > .env environment variable
    hf_token = hf_token or os.getenv("HF_TOKEN")

    audio_proc = AudioProcessor(output_dir=project_dir)
    transcriber = Transcriber(device=device, hf_token=hf_token)
    translator = Translator(
        ollama_url=ollama_url, model=ollama_model, audio_model=ollama_audio_model
    )
    synthesizer = Synthesizer(engine=engine, output_dir=audio_segments_dir, device=device)

    # 2. Process Audio
    orig_audio = temp_dir / "original_audio.wav"
    if not orig_audio.exists():
        orig_audio = audio_proc.extract_audio(video_path)
    else:
        logger.info(f"Skipping audio extraction, using existing: {orig_audio}")

    # Demucs output paths
    vocals = temp_dir / "htdemucs" / "original_audio" / "vocals.wav"
    background = temp_dir / "htdemucs" / "original_audio" / "no_vocals.wav"

    if not vocals.exists() or not background.exists():
        vocals, background = audio_proc.separate_vocals(orig_audio)
    else:
        logger.info(f"Skipping vocal separation, using existing: {vocals}")

    # Parallel HQ track for XTTS reference cloning (44.1kHz stereo).
    orig_audio_hq = temp_dir / "original_audio_hq.wav"
    if not orig_audio_hq.exists():
        orig_audio_hq = audio_proc.extract_audio_hq(video_path)
    else:
        logger.info(f"Skipping HQ audio extraction, using existing: {orig_audio_hq}")

    vocals_hq = temp_dir / "htdemucs" / "original_audio_hq" / "vocals.wav"
    if not vocals_hq.exists():
        vocals_hq, _ = audio_proc.separate_vocals(orig_audio_hq)
    else:
        logger.info(f"Skipping HQ vocal separation, using existing: {vocals_hq}")

    # 3. Transcribe & Diarize
    transcript_path = project_dir / "transcript.json"
    if not transcript_path.exists():
        transcript = transcriber.transcribe(vocals)
        transcriber.save_transcript(transcript, transcript_path)
    else:
        logger.info(f"Skipping transcription, using existing: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)

    # 4. Translate
    translated_transcript_path = project_dir / "transcript_translated.json"
    if not translated_transcript_path.exists():
        # Pre-compute expanded placement windows so short segments get
        # enough room for TTS to speak the translation at natural pace.
        audio_end_sec = len(AudioSegment.from_wav(orig_audio)) / 1000.0
        annotate_effective_windows(transcript["segments"], audio_end=audio_end_sec)

        # Opportunistic: if the source video has a target-language subtitle
        # stream, extract it as a translation hint for Gemma.
        subtitle_entries = None
        srt_path = audio_proc.extract_target_subtitles(video_path, target_lang)
        if srt_path and srt_path.exists():
            subtitle_entries = parse_srt(srt_path)
            logger.info(
                f"Loaded {len(subtitle_entries)} {target_lang} subtitle entries as reconciliation hints."
            )

        translated_segments = translator.translate_segments_multimodal(
            transcript["segments"],
            vocals_path=vocals,
            target_lang=target_lang,
            subtitle_entries=subtitle_entries,
        )
        transcript["translated_segments"] = translated_segments
        transcriber.save_transcript(transcript, translated_transcript_path)
    else:
        logger.info(f"Skipping translation, using existing: {translated_transcript_path}")
        with open(translated_transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
        translated_segments = transcript["translated_segments"]

    # 5. Extract Speaker References
    synthesizer.ref_audio_dir = project_dir / "references"
    synthesizer.ref_audio_dir.mkdir(parents=True, exist_ok=True)
    references = synthesizer.extract_speaker_references(
        vocals, transcript, hq_vocals_path=vocals_hq
    )

    # 6. Synthesize & Place Audio
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
        "Hindi": "hi",
    }
    tts_lang = lang_map.get(target_lang, "en")

    original_audio = AudioSegment.from_wav(orig_audio)
    dubbed_audio_track = AudioSegment.silent(duration=len(original_audio))

    for i, segment in enumerate(translated_segments):
        speaker = segment.get("speaker")
        text = segment.get("text")
        # Use the widened placement window if present; fall back to the
        # raw diarized boundaries for backward compatibility.
        start_time = segment.get("effective_start", segment.get("start"))
        end_time = segment.get("effective_end", segment.get("end"))

        if not text:
            continue

        if not speaker or speaker not in references:
            if references:
                speaker = list(references.keys())[0]
                logger.warning(
                    f"Speaker missing or no references for segment {i}. Falling back to speaker {speaker}."
                )
            else:
                logger.warning(f"No references available at all, skipping segment {i}.")
                continue

        clip_name = f"segment_{i}_{speaker}.wav"
        clip_path = audio_segments_dir / clip_name

        emotion_tag = segment.get("emotion", "[NEUTRAL]")
        if not emotion_tag.startswith("["):
            emotion_tag = f"[{emotion_tag}"
        if not emotion_tag.endswith("]"):
            emotion_tag = f"{emotion_tag}]"
        emotion_tag = emotion_tag.upper()

        clean_text = text.replace(emotion_tag, "").strip()

        if not clip_path.exists():
            clip_path = synthesizer.synthesize(
                clean_text,
                speaker,
                references[speaker],
                clip_name,
                language=tts_lang,
                emotion=emotion_tag,
            )
        else:
            logger.info(f"Skipping synthesis for segment {i}, using existing: {clip_path}")

        if not clip_path or not os.path.exists(clip_path):
            logger.warning(f"Skipping segment {i} (Synthesized file not found)")
            continue

        target_duration = end_time - start_time
        synthesizer.adjust_speed(clip_path, target_duration)

        segment_audio = AudioSegment.from_wav(clip_path)
        
        # Apply a short fade to eliminate pops/clicks at the segment boundaries
        fade_duration = min(5, len(segment_audio) // 2)
        if fade_duration > 0:
            segment_audio = segment_audio.fade_in(fade_duration).fade_out(fade_duration)

        start_ms = int(start_time * 1000)
        dubbed_audio_track = dubbed_audio_track.overlay(segment_audio, position=start_ms)

    # 7. Mix with background and Remux
    video_output_path = project_dir / f"dubbed_{Path(video_path).name}"

    logger.info("Mixing final audio track...")
    background_audio = AudioSegment.from_wav(background)
    dubbed_audio_track = audio_proc.match_loudness(dubbed_audio_track, vocals)
    final_mixed_audio = audio_proc.duck_audio(dubbed_audio_track, background_audio)

    final_audio_path = project_dir / "final_audio.wav"
    final_mixed_audio.export(final_audio_path, format="wav")

    logger.info(f"Remuxing final video to {video_output_path}...")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(final_audio_path),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(video_output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg remuxing failed: {e.stderr if e.stderr else str(e)}")
        raise RuntimeError(f"FFmpeg failed during remuxing: {e.stderr if e.stderr else str(e)}") from e

    logger.info(f"Auto-Dubbing Complete! Output: {video_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Dub Pipeline")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--lang", default="Spanish", help="Target language for translation")
    parser.add_argument("--hf_token", help="Hugging Face token for diarization")
    parser.add_argument("--device", help="Device to use (cuda, mps, cpu)")
    parser.add_argument("--ollama_url", help="Ollama instance URL")
    parser.add_argument(
        "--ollama_model", help="Ollama model name for text translation (e.g. gemma4:26b)"
    )
    parser.add_argument(
        "--ollama_audio_model",
        help="Ollama model for audio-informed emotion tagging (e.g. gemma4:e4b)",
    )
    parser.add_argument(
        "--engine", default="xtts", choices=["xtts", "fish"], help="Synthesis engine to use"
    )

    args = parser.parse_args()
    main(
        args.video,
        target_lang=args.lang,
        hf_token=args.hf_token,
        device=args.device,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        ollama_audio_model=args.ollama_audio_model,
        engine=args.engine,
    )
