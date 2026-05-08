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
    **kwargs,
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

    # Separate vocals (single HQ run for everything)
    vocals = temp_dir / "htdemucs" / "original_audio" / "vocals.wav"
    background = temp_dir / "htdemucs" / "original_audio" / "no_vocals.wav"

    if not vocals.exists() or not background.exists():
        vocals, background = audio_proc.separate_vocals(orig_audio)
    else:
        logger.info(f"Skipping vocal separation, using existing: {vocals}")

    # 3. Transcribe & Diarize
    transcript_path = project_dir / "transcript.json"
    if not transcript_path.exists():
        # Apply Vocal Focus filter to clean up isolated audio for better diarization
        vocals_focused = audio_proc.focus_vocals(vocals)

        transcript = transcriber.transcribe(
            vocals_focused,
            min_speakers=kwargs.get("min_speakers"),
            max_speakers=kwargs.get("max_speakers"),
        )
        transcriber.save_transcript(transcript, transcript_path)

        unique_speakers = sorted(
            {s.get("speaker") for s in transcript["segments"] if s.get("speaker")}
        )
        logger.info(
            f"Diarization complete. Found {len(unique_speakers)} speakers: {', '.join(unique_speakers)}"
        )
        if len(unique_speakers) <= 1 and not (
            kwargs.get("min_speakers") or kwargs.get("max_speakers")
        ):
            logger.warning(
                "Only one speaker detected. If this is incorrect, try running with --min_speakers."
            )

        # Cleanup focused file to save space
        if vocals_focused.exists() and vocals_focused != vocals:
            vocals_focused.unlink()
    else:
        logger.info(f"Skipping transcription, using existing: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            transcript = json.load(f)
        unique_speakers = sorted(
            {s.get("speaker") for s in transcript["segments"] if s.get("speaker")}
        )
        logger.info(
            f"Loaded transcript with {len(unique_speakers)} detected speakers: {', '.join(unique_speakers)}"
        )

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
    # Increase to 5 clips and 5s min duration for a more robust vocal profile
    references = synthesizer.extract_speaker_references(
        vocals, transcript, target_clips=5, min_duration=5
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

    # Pre-load original vocals for song preservation handling
    original_vocals = AudioSegment.from_wav(vocals)

    logger.info(
        f"Starting synthesis for {len(translated_segments)} segments using {engine} engine..."
    )

    # Sort segments by start time to ensure logical processing and overlap detection
    translated_segments.sort(key=lambda x: x.get("effective_start", x.get("start", 0)))

    for i, segment in enumerate(translated_segments):
        speaker = segment.get("speaker")
        text = segment.get("text")

        # Check for overlaps for logging
        if i < len(translated_segments) - 1:
            next_seg = translated_segments[i + 1]
            curr_end = segment.get("effective_end", segment.get("end", 0))
            next_start = next_seg.get("effective_start", next_seg.get("start", 0))
            if curr_end > next_start:
                logger.info(
                    f"Overlap detected: Segment {i} ({segment.get('speaker')}) ends at {curr_end:.2f}s, Segment {i+1} ({next_seg.get('speaker')}) starts at {next_start:.2f}s"
                )

        # Log progress every 5 segments
        if i % 5 == 0 or i == len(translated_segments) - 1:
            logger.info(
                f"Processing segment {i+1}/{len(translated_segments)} (Speaker: {speaker})..."
            )

        # Use the widened placement window if present; fall back to the
        # raw diarized boundaries for backward compatibility.
        start_time = segment.get("effective_start", segment.get("start"))
        end_time = segment.get("effective_end", segment.get("end"))

        if not text:
            continue

        # Song handling: Preservation of original singing
        if segment.get("is_song"):
            logger.info(f"Segment {i} identified as SONG. Preserving original vocals.")
            # Extract the original vocals for this time range
            start_ms = int(segment.get("start") * 1000)
            end_ms = int(segment.get("end") * 1000)
            song_clip = original_vocals[start_ms:end_ms]

            start_ms_dub = int(start_time * 1000)
            dubbed_audio_track = dubbed_audio_track.overlay(song_clip, position=start_ms_dub)
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
            # Add a language hint to help Fish Speech reduce cross-lingual accent
            # We add [western accent] for English to nudge it away from rigid cadence
            lang_hint = f"[{target_lang.lower()}]"
            if target_lang.lower() == "english":
                lang_hint = "[english] [western accent]"

            full_text = f"{lang_hint} {clean_text}" if engine == "fish" else clean_text

            clip_path = synthesizer.synthesize(
                full_text,
                speaker,
                references[speaker],
                clip_name,
                language=tts_lang,
                emotion=emotion_tag,
                temp=kwargs.get("tts_temp", 0.7),
                top_p=kwargs.get("tts_top_p", 0.8),
            )

            # Apply RVC Skin (Identity Transfer) if a model exists for this speaker
            if clip_path:
                clip_path = synthesizer.apply_rvc(clip_path, speaker)
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
    # Force .mp4 output for maximum compatibility across players
    video_output_path = project_dir / f"dubbed_{Path(video_path).stem}.mp4"

    logger.info("Mixing final audio track...")
    background_audio = AudioSegment.from_wav(background)
    dubbed_audio_track = audio_proc.match_loudness(dubbed_audio_track, vocals)
    final_mixed_audio = audio_proc.duck_audio(dubbed_audio_track, background_audio)

    final_audio_path = project_dir / "final_audio.wav"
    final_mixed_audio.export(final_audio_path, format="wav")

    logger.info(f"Remuxing final video to {video_output_path}...")

    # Choose audio codec based on extension
    audio_codec = "aac"
    if video_output_path.suffix.lower() == ".webm":
        audio_codec = "libopus"

    try:
        # Determine source language ISO
        src_iso = audio_proc.LANG_TO_ISO3.get(transcript.get("language", "Japanese"), "und")
        tgt_iso = audio_proc.LANG_TO_ISO3.get(target_lang, "eng")

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-i",
                str(final_audio_path),
                "-map",
                "0:v",  # Map original video
                "-map",
                "0:a",  # Map original audio (Track 1)
                "-map",
                "1:a",  # Map dubbed audio (Track 2)
                "-c:v",
                "copy",
                "-c:a:0",
                "copy",  # Keep original audio as-is
                "-c:a:1",
                audio_codec,  # Encode dubbed audio
                "-metadata:s:a:0",
                f"language={src_iso}",
                "-metadata:s:a:0",
                "title=Original",
                "-metadata:s:a:1",
                f"language={tgt_iso}",
                "-metadata:s:a:1",
                "title=Dubbed",
                "-disposition:a:0",
                "0",  # Original is NOT default
                "-disposition:a:1",
                "default",  # Dub is default
                "-shortest",
                str(video_output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg remuxing failed: {e.stderr if e.stderr else str(e)}")
        raise RuntimeError(
            f"FFmpeg failed during remuxing: {e.stderr if e.stderr else str(e)}"
        ) from e

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
    parser.add_argument("--min_speakers", type=int, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", type=int, help="Maximum number of speakers")
    parser.add_argument("--tts_temp", type=float, default=0.7, help="TTS Temperature (0.1-1.0)")
    parser.add_argument("--tts_top_p", type=float, default=0.8, help="TTS Top-P sampling")

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
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        tts_temp=args.tts_temp,
        tts_top_p=args.tts_top_p,
    )
