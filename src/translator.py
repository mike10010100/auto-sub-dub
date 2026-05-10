import io
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pyphen
from ollama import Client
from tqdm import tqdm

logger = logging.getLogger(__name__)

VALID_EMOTIONS = {
    "[NEUTRAL]",
    "[WHISPER]",
    "[ANGRY]",
    "[EXCITED]",
    "[SAD]",
    "[SARCASM]",
    "[FRIENDLY]",
    "[SHOUTING]",
}

# Typical spoken-word rates used only for length budgeting, not precise timing.
WORDS_PER_SECOND = {
    "English": 2.5,
    "Spanish": 2.8,
    "French": 3.2,
    "German": 2.1,
    "Italian": 3.0,
    "Portuguese": 2.7,
    "Polish": 2.4,
    "Turkish": 2.4,
    "Russian": 2.3,
    "Dutch": 2.3,
    "Czech": 2.3,
    "Hungarian": 2.3,
    "Hindi": 2.8,
    "Arabic": 2.4,
}
# CJK languages use character counts instead.
CHARS_PER_SECOND = {"Chinese": 5.5, "Japanese": 7.0, "Korean": 6.5}


def _estimate_spoken_duration(text, language):
    if language in CHARS_PER_SECOND:
        return len(text.strip()) / CHARS_PER_SECOND[language]
    words = max(1, len(text.split()))
    return words / WORDS_PER_SECOND.get(language, 2.5)


def _length_budget(target_duration, language, headroom=1.0):
    """Max words (or chars for CJK) that fit in target_duration * headroom."""
    budget_sec = target_duration * headroom
    if language in CHARS_PER_SECOND:
        return max(1, int(CHARS_PER_SECOND[language] * budget_sec))
    return max(1, int(WORDS_PER_SECOND.get(language, 2.5) * budget_sec))


def _count_syllables(text, language):
    """Estimate syllable count for timing-aware translation."""
    if not text:
        return 0
    # CJK: 1 character approx 1 syllable/mora
    if language in ["Chinese", "Japanese", "Korean"]:
        return len(re.sub(r"[^\w]", "", text))

    # English/European: use Pyphen
    lang_code = {
        "English": "en_US",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
    }.get(language, "en_US")

    try:
        dic = pyphen.Pyphen(lang=lang_code)
    except Exception:
        dic = pyphen.Pyphen(lang="en_US")

    count = 0
    for word in text.split():
        word = re.sub(r"[^\w]", "", word)
        if not word:
            continue
        # Pyphen returns hyphenated word, count parts
        hyphenated = dic.inserted(word)
        count += len(hyphenated.split("-"))
    return count


class Translator:
    """
    Two-model pipeline:
      - audio_model (e.g. gemma4:e4b): audio-informed emotion tagging.
        Audio is delivered via the `images` field as a 16 kHz mono WAV,
        which Ollama routes to the audio path via RIFF/WAVE magic-byte
        detection (see ollama/ollama#11798). Only the E-variants of
        Gemma 4 accept audio.
      - model (e.g. gemma4:26b): text-only translation, conditioned on
        the emotion tag emitted by the audio pass.
    """

    def __init__(self, ollama_url=None, model=None, audio_model=None):  # pragma: no cover
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4:26b")
        self.audio_model = audio_model or os.getenv("OLLAMA_AUDIO_MODEL", "gemma4:e4b")

        self.client = Client(host=self.ollama_url)
        logger.info(
            f"Translator: ollama={self.ollama_url} "
            f"translate={self.model} audio={self.audio_model}"
        )

    # Gemma 4 e4b's audio encoder crashes the runner on clips shorter than
    # ~1500 ms. Pad to a safe floor with trailing silence.
    MIN_AUDIO_MS = 2000

    @classmethod
    def _clip_to_wav_bytes(cls, clip):  # pragma: no cover  (pydub/audio IO)
        # Ollama's audio-via-images path requires 16 kHz mono WAV with a
        # full RIFF header. Raw PCM fails silently.
        from pydub import AudioSegment

        mono_16k = clip.set_frame_rate(16000).set_channels(1)
        if len(mono_16k) < cls.MIN_AUDIO_MS:
            mono_16k = mono_16k + AudioSegment.silent(
                duration=cls.MIN_AUDIO_MS - len(mono_16k),
                frame_rate=16000,
            )
        buf = io.BytesIO()
        mono_16k.export(buf, format="wav")
        return buf.getvalue()

    def _chat_with_retry(
        self, *, model, messages, options, attempts=3
    ):  # pragma: no cover  (Ollama IO)
        # Audio inference on Ollama is currently crash-prone (see #15333),
        # so retry with linear backoff.
        last_err = None
        for i in range(attempts):
            try:
                return self.client.chat(model=model, messages=messages, options=options)
            except Exception as e:
                last_err = e
                logger.warning(f"Ollama call failed (attempt {i+1}/{attempts}): {e}")
                time.sleep(1.0 * (i + 1))
        raise last_err

    def _tag_emotion(self, audio_bytes, original_text):  # pragma: no cover
        """Audio-informed emotion tag. Returns one of VALID_EMOTIONS."""
        system_prompt = (
            "You classify the emotional delivery of a short speech clip. "
            "Listen to the audio and pick exactly one tag from this set: "
            "[NEUTRAL] [WHISPER] [ANGRY] [EXCITED] [SAD] [SARCASM] [FRIENDLY] [SHOUTING]. "
            "Respond with ONLY the tag, including the brackets. No other text."
        )
        try:
            resp = self._chat_with_retry(
                model=self.audio_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        # `images` first so audio precedes text, per the
                        # Ollama audio-via-images workaround guidance.
                        "images": [audio_bytes],
                        "content": f"Transcript for reference: '{original_text}'. Output only the tag.",
                    },
                ],
                options={
                    "temperature": 0.2,
                    "num_predict": 16,
                    "num_ctx": 8192,  # audio embeddings OOM above this
                },
            )
            tag = resp["message"]["content"].strip().upper()
            # Tolerate models that wrap or add prose.
            for candidate in VALID_EMOTIONS:
                if candidate in tag:
                    return candidate
            return "[NEUTRAL]"
        except Exception as e:
            logger.error(f"Emotion tagging failed, defaulting to [NEUTRAL]: {e}")
            return "[NEUTRAL]"

    @staticmethod
    def _format_context(segments, start_i, end_i, current_idx, completed=None):
        """Render context window as numbered lines, marking the target line."""
        if end_i <= start_i:
            return ""
        lines = []
        for j in range(start_i, end_i):
            seg = segments[j]
            src = (seg.get("text") or "").strip()
            speaker = seg.get("speaker") or "?"
            if j == current_idx:
                lines.append(f"  [{j}] ({speaker}) >>> {src}  ← TRANSLATE THIS LINE")
            else:
                # Prefer already-translated output when available so the
                # model can keep pronouns/register consistent.
                done = (completed or {}).get(j)
                if done:
                    lines.append(f"  [{j}] ({speaker}) {src}  → {done}")
                else:
                    lines.append(f"  [{j}] ({speaker}) {src}")
        return "\n".join(lines)

    @staticmethod
    def _subs_for_segment(seg, subtitle_entries, min_overlap=0.3):
        """
        Return subtitle entries that overlap the diarized segment by at least
        `min_overlap` seconds. Subs are a reference hint; we want to surface
        the candidates and let Gemma pick what matters.
        """
        if not subtitle_entries:
            return []
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        hits = []
        for entry in subtitle_entries:
            overlap = min(seg_end, entry["end"]) - max(seg_start, entry["start"])
            if overlap >= min_overlap:
                hits.append(entry)
        return hits

    def _translate_once(
        self,
        original_text,
        emotion,
        duration,
        target_lang,
        budget,  # pragma: no cover
        stricter=False,
        context_block="",
        subtitle_hint="",
        target_syllables=None,
    ):
        unit = "characters" if target_lang in CHARS_PER_SECOND else "words"
        extra = (
            " Your previous attempt was too long. Be MORE concise this time, "
            "drop filler, and keep only essential meaning."
            if stricter
            else ""
        )
        ctx_rule = (
            "5. Use the conversation context to disambiguate meaning. Resolve "
            "greetings vs. farewells, pronouns, gendered agreement, callbacks, "
            "and sarcasm from the surrounding lines. Translate ONLY the line "
            "marked '← TRANSLATE THIS LINE'.\n"
            if context_block
            else ""
        )
        sub_rule = (
            "6. A professional subtitle translation is provided as REFERENCE. "
            "It may be condensed or paraphrased for reading speed, so treat it "
            "as a strong hint for meaning and terminology, but prefer a natural "
            "spoken rendering of the full source line over a terse subtitle if "
            "the budget allows. Do not copy the subtitle verbatim.\n"
            if subtitle_hint
            else ""
        )

        syllable_rule = ""
        if target_syllables:
            syllable_rule = (
                f"7. CRITICAL TIMING: The source line had {target_syllables} syllables. "
                f"Your translation MUST be between {max(1, int(target_syllables*0.8))} and "
                f"{int(target_syllables*1.2)} syllables to match the speaker's mouth movements.\n"
            )

        system_prompt = (
            "<|think|>You are a professional video translator and voice director. "
            f"Translate subtitles into {target_lang}.\n\n"
            "RULES:\n"
            f"1. Output natural {target_lang} that preserves speaker intent.\n"
            f"2. The translation MUST be speakable within {duration:.2f}s. "
            f"HARD LIMIT: no more than {budget} {unit}. Cut filler, interjections, "
            "and redundancy before exceeding this limit.\n"
            "3. Match the delivery implied by the provided emotion tag.\n"
            "4. Identify if the line is part of a SONG (opening/closing theme or background music). "
            "Songs often have repetitive structures, rhyming, or poetic flow. Set 'is_song': true if so.\n"
            "5. Respond with ONLY a JSON object (after any thinking block): "
            '{"translated_text": "...", "is_song": true/false}\n'
            + ctx_rule
            + sub_rule
            + syllable_rule
            + extra
        )
        context_section = f"CONVERSATION CONTEXT:\n{context_block}\n\n" if context_block else ""
        sub_section = (
            f"SUBTITLE REFERENCE ({target_lang}):\n{subtitle_hint}\n\n" if subtitle_hint else ""
        )
        user_msg = (
            f"{context_section}"
            f"{sub_section}"
            f"Emotion: {emotion}. Target duration: {duration:.2f}s. "
            f"Budget: {budget} {unit}. Translate line: '{original_text}'"
        )
        resp = self._chat_with_retry(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            options={"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        )
        content = resp["message"]["content"].strip()
        if "<channel|>" in content:
            content = content.split("<channel|>")[-1].strip()
        try:
            if "{" in content and "}" in content:
                content = content[content.find("{") : content.rfind("}") + 1]
            data = json.loads(content)
            return (data.get("translated_text", "") or "").strip(), bool(data.get("is_song", False))
        except (json.JSONDecodeError, ValueError):
            return content.strip(), False

    def _translate_text(
        self,
        original_text,
        emotion,
        duration,
        target_lang,  # pragma: no cover
        context_block="",
        subtitle_hint="",
        overrun_ratio=1.2,
        source_lang="Japanese",
    ):
        """Translate; if the result wouldn't fit in the source window, retry once with a tighter budget."""
        budget = _length_budget(duration, target_lang, headroom=1.0)

        # Calculate source syllables for target-matching
        source_syllables = _count_syllables(original_text, source_lang)

        translated, is_song = self._translate_once(
            original_text,
            emotion,
            duration,
            target_lang,
            budget,
            context_block=context_block,
            subtitle_hint=subtitle_hint,
            target_syllables=source_syllables,
        )
        if not translated:
            return original_text, is_song

        if is_song:
            return translated, is_song

        est = _estimate_spoken_duration(translated, target_lang)
        if est <= duration * overrun_ratio:
            return translated, is_song

        new_budget = max(1, int(budget * 0.8))
        # If we can't shrink the budget further, retrying is pointless.
        if new_budget >= budget or budget <= 1:
            return translated, is_song

        logger.info(
            f"Retry (overrun): est {est:.2f}s vs target {duration:.2f}s — "
            f"shrinking budget {budget} → {new_budget}"
        )
        retry, is_song_retry = self._translate_once(
            original_text,
            emotion,
            duration,
            target_lang,
            budget=new_budget,
            stricter=True,
            context_block=context_block,
            subtitle_hint=subtitle_hint,
            target_syllables=source_syllables,
        )
        if retry and _estimate_spoken_duration(retry, target_lang) < est:
            return retry, is_song_retry or is_song
        return translated, is_song

    def review_diarization(self, segments):  # pragma: no cover
        """
        Use the LLM to review conversational flow and logically correct
        speaker diarization mistakes (e.g., merging overlapping characters).
        """
        if not segments:
            return segments

        logger.info(f"Starting Semantic Diarization Review for {len(segments)} segments...")

        chunk_size = 25
        out_segments = [s.copy() for s in segments]

        for i in tqdm(range(0, len(out_segments), chunk_size), desc="Semantic Review"):
            chunk = out_segments[i : i + chunk_size]

            # Format the chunk
            lines = []
            for j, seg in enumerate(chunk):
                src = (seg.get("text") or "").strip()
                speaker = seg.get("speaker") or "?"
                lines.append(f"[{i+j}] ({speaker}): {src}")

            context_block = "\n".join(lines)

            system_prompt = (
                "<|think|>You are a logic analyzer. Review the following dialogue transcript. "
                "The acoustic diarization AI often misidentifies speakers when their voices are similar or overlapping. "
                "Look for logical breaks in conversational flow (e.g., a person answering their own question, "
                "or a sudden shift in tone/topic attributed to the same speaker). "
                "If you find an error, correct the speaker label to match the logical flow of the conversation. "
                "Return ONLY a JSON object containing an array of corrections (after your thinking block). "
                'Format: {"corrections": [{"index": 12, "new_speaker": "SPEAKER_02"}]}\n'
                'If no corrections are needed, return an empty array: {"corrections": []}.'
            )

            try:
                resp = self._chat_with_retry(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"TRANSCRIPT CHUNK:\n{context_block}"},
                    ],
                    options={"temperature": 0.2, "top_p": 0.95},
                )

                content = resp["message"]["content"].strip()
                if "<channel|>" in content:
                    content = content.split("<channel|>")[-1].strip()
                if "{" in content and "}" in content:
                    content = content[content.find("{") : content.rfind("}") + 1]

                data = json.loads(content)
                corrections = data.get("corrections", [])

                for corr in corrections:
                    idx = corr.get("index")
                    new_spk = corr.get("new_speaker")
                    if (
                        idx is not None
                        and new_spk
                        and isinstance(idx, int)
                        and i <= idx < i + chunk_size
                    ):
                        old_spk = out_segments[idx].get("speaker")
                        if old_spk != new_spk:
                            logger.info(
                                f"  Semantic Review Fixed: Segment {idx} ({old_spk} -> {new_spk})"
                            )
                            out_segments[idx]["speaker"] = new_spk

            except Exception as e:
                logger.warning(f"Semantic Diarization Review failed for chunk {i}: {e}")
                continue

        return out_segments

    def translate_segments_multimodal(  # pragma: no cover  (orchestrates Ollama + audio)
        self,
        segments,
        vocals_path,
        target_lang="English",
        max_workers=2,
        context_before=5,
        context_after=3,
        subtitle_entries=None,
        source_lang="Japanese",
    ):
        """
        Two-pass translation:
          1. Emotion tags via gemma4:e4b (parallel — independent audio calls).
          2. Text translation via gemma4:26b sequentially, feeding each call
             a rolling window of the surrounding source lines plus the
             already-translated lines so the model can disambiguate greetings
             vs. farewells, pronouns, sarcasm, and callbacks.
        """
        from pydub import AudioSegment

        sub_count = len(subtitle_entries) if subtitle_entries else 0
        logger.info(
            f"Translating {len(segments)} segments to {target_lang} "
            f"(audio-tag={self.audio_model}, translate={self.model}, "
            f"ctx=-{context_before}/+{context_after}, subs={sub_count})"
        )
        audio = AudioSegment.from_wav(vocals_path)

        # --- Pass 1: audio-informed emotion tagging (parallel). ---
        def tag_one(idx_seg):
            idx, segment = idx_seg
            original_text = (segment.get("text") or "").strip()
            if not original_text:
                return idx, "[NEUTRAL]"
            clip = audio[int(segment.get("start", 0) * 1000) : int(segment.get("end", 0) * 1000)]
            audio_bytes = self._clip_to_wav_bytes(clip)
            return idx, self._tag_emotion(audio_bytes, original_text)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tag_results = list(
                tqdm(
                    executor.map(tag_one, list(enumerate(segments))),
                    total=len(segments),
                    desc="Tagging emotion (audio)",
                )
            )
        emotions = {idx: tag for idx, tag in tag_results}

        # --- Pass 2: text translation sequentially with rolling context. ---
        completed = {}  # idx -> translated text, feeds into context of later calls
        out = [None] * len(segments)

        for idx, segment in enumerate(tqdm(segments, desc="Translating (text+context)")):
            original_text = (segment.get("text") or "").strip()
            if not original_text:
                out[idx] = segment
                continue

            eff_start = segment.get("effective_start", segment.get("start", 0.0))
            eff_end = segment.get("effective_end", segment.get("end", 0.0))
            duration = max(0.0, eff_end - eff_start)
            emotion = emotions.get(idx, "[NEUTRAL]")

            ctx_lo = max(0, idx - context_before)
            ctx_hi = min(len(segments), idx + context_after + 1)
            context_block = self._format_context(segments, ctx_lo, ctx_hi, idx, completed)

            sub_hits = self._subs_for_segment(segment, subtitle_entries or [])
            subtitle_hint = " ".join(h["text"] for h in sub_hits).strip()

            try:
                translated_text, is_song = self._translate_text(
                    original_text,
                    emotion,
                    duration,
                    target_lang,
                    context_block=context_block,
                    subtitle_hint=subtitle_hint,
                    source_lang=source_lang,
                )
            except Exception as e:
                logger.error(f"Translation failed for segment {idx}: {e}")
                out[idx] = segment
                continue

            completed[idx] = translated_text
            new_segment = segment.copy()
            new_segment["original_text"] = original_text
            new_segment["text"] = translated_text
            new_segment["emotion"] = emotion
            new_segment["is_song"] = is_song
            out[idx] = new_segment

        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Translator module loaded.")
