import pytest

from src.translator import (
    CHARS_PER_SECOND,
    VALID_EMOTIONS,
    WORDS_PER_SECOND,
    Translator,
    _count_syllables,
    _estimate_spoken_duration,
    _length_budget,
)


def test_count_syllables():
    # English
    assert _count_syllables("hello", "English") == 2
    assert _count_syllables("supermarket", "English") == 4

    # Japanese (mora counting)
    assert _count_syllables("こんにちは", "Japanese") == 5
    assert _count_syllables("テレビ", "Japanese") == 3

    # Empty
    assert _count_syllables("", "English") == 0


def test_estimate_spoken_duration_english_words():
    # 5 words at 2.5 wps → 2.0s
    assert _estimate_spoken_duration("one two three four five", "English") == pytest.approx(2.0)


def test_estimate_spoken_duration_cjk_chars():
    # 11 chars at 7.0 cps → ~1.57s (Japanese)
    assert _estimate_spoken_duration("こんにちは世界！テスト", "Japanese") == pytest.approx(
        11 / 7.0
    )


def test_estimate_spoken_duration_empty_gets_floor():
    # Min 1 word.
    assert _estimate_spoken_duration("", "English") == pytest.approx(1 / 2.5)


def test_estimate_spoken_duration_unknown_language_uses_default():
    # Unknown language falls back to 2.5 wps.
    assert _estimate_spoken_duration("one two", "Klingon") == pytest.approx(2 / 2.5)


def test_length_budget_scales_with_duration():
    # Integer-flooring means the exact ratio can drift by 1; assert monotonicity
    # and that doubling the duration roughly doubles the word budget.
    b1 = _length_budget(1.0, "English")
    b2 = _length_budget(2.0, "English")
    b4 = _length_budget(4.0, "English")
    assert b1 < b2 < b4
    # int-floor adds up to 1 unit of drift per call (wps=2.5 → 2,5,10),
    # so require approximate — not exact — linear scaling.
    assert b2 == int(2.5 * 2)
    assert b4 == int(2.5 * 4)


def test_length_budget_cjk_uses_chars():
    assert _length_budget(1.0, "Chinese") == int(CHARS_PER_SECOND["Chinese"])


def test_length_budget_returns_positive_min():
    assert _length_budget(0.0, "English") >= 1


def test_format_context_empty_when_no_range():
    assert Translator._format_context([], 0, 0, 0) == ""


def test_format_context_marks_target_line():
    segs = [
        {"text": "line zero", "speaker": "A"},
        {"text": "line one", "speaker": "B"},
        {"text": "line two", "speaker": "A"},
    ]
    out = Translator._format_context(segs, 0, 3, current_idx=1)
    lines = out.splitlines()
    assert "← TRANSLATE THIS LINE" in lines[1]
    assert "← TRANSLATE THIS LINE" not in lines[0]
    assert "← TRANSLATE THIS LINE" not in lines[2]
    assert "(A)" in lines[0]
    assert "(B)" in lines[1]


def test_format_context_uses_completed_translations():
    segs = [{"text": "bonjour", "speaker": "A"}, {"text": "au revoir", "speaker": "A"}]
    completed = {0: "hello"}
    out = Translator._format_context(segs, 0, 2, current_idx=1, completed=completed)
    assert "→ hello" in out
    # Current target line is never shown with completed translation.
    assert "→" not in out.splitlines()[1]


def test_format_context_handles_missing_speaker_and_text():
    segs = [{"text": None, "speaker": None}]
    out = Translator._format_context(segs, 0, 1, current_idx=0)
    assert "(?)" in out


def test_subs_for_segment_finds_overlap():
    seg = {"start": 1.0, "end": 3.0}
    entries = [
        {"start": 0.0, "end": 0.5, "text": "before"},  # no overlap
        {"start": 0.9, "end": 2.0, "text": "partial"},  # 1.0s overlap
        {"start": 2.5, "end": 4.0, "text": "tail"},  # 0.5s overlap
        {"start": 10.0, "end": 11.0, "text": "after"},  # no overlap
    ]
    hits = Translator._subs_for_segment(seg, entries)
    texts = [h["text"] for h in hits]
    assert "before" not in texts
    assert "partial" in texts
    assert "tail" in texts
    assert "after" not in texts


def test_subs_for_segment_respects_min_overlap():
    seg = {"start": 1.0, "end": 3.0}
    entries = [{"start": 0.9, "end": 1.1, "text": "tiny"}]  # 0.1s overlap
    assert Translator._subs_for_segment(seg, entries, min_overlap=0.3) == []
    assert Translator._subs_for_segment(seg, entries, min_overlap=0.05)[0]["text"] == "tiny"


def test_subs_for_segment_empty_entries():
    assert Translator._subs_for_segment({"start": 0, "end": 1}, []) == []
    assert Translator._subs_for_segment({"start": 0, "end": 1}, None) == []


def test_valid_emotions_set_nonempty():
    assert "[NEUTRAL]" in VALID_EMOTIONS
    assert len(VALID_EMOTIONS) >= 4


def test_words_per_second_has_major_languages():
    for lang in ("English", "Spanish", "French", "German"):
        assert lang in WORDS_PER_SECOND


def test_translate_once_fallback():
    from unittest.mock import MagicMock

    translator = Translator(ollama_url="http://localhost:11434")

    # Mock response format:
    # First response: empty content (truncated reasoning)
    resp1 = {"message": {"content": ""}}
    # Second response: fallback content
    resp2 = {"message": {"content": '{"translated_text": "Hello", "is_song": false}'}}

    translator._chat_with_retry = MagicMock(side_effect=[resp1, resp2])

    res, is_song = translator._translate_once(
        original_text="こんにちは",
        emotion="[NEUTRAL]",
        duration=5.0,
        target_lang="English",
        budget=10,
        think_translation=True,
    )

    assert res == "Hello"
    assert is_song is False
    assert translator._chat_with_retry.call_count == 2

    # Check that the second call used think=False
    args, kwargs = translator._chat_with_retry.call_args_list[1]
    assert kwargs.get("think") is False


def test_review_diarization_fallback():
    from unittest.mock import MagicMock

    translator = Translator(ollama_url="http://localhost:11434")

    # First response: empty content
    resp1 = {"message": {"content": ""}}
    # Second response: fallback corrections JSON
    resp2 = {
        "message": {
            "content": '<reasoning>ok</reasoning>```json\n{"corrections": [{"index": 0, "new_speaker": "SPEAKER_02"}]}\n```'
        }
    }

    translator._chat_with_retry = MagicMock(side_effect=[resp1, resp2])

    segments = [
        {"text": "Hello", "speaker": "SPEAKER_01"},
        {"text": "Hi", "speaker": "SPEAKER_02"},
    ]
    reviewed = translator.review_diarization(segments)

    assert reviewed[0]["speaker"] == "SPEAKER_02"
    assert translator._chat_with_retry.call_count == 2

    args, kwargs = translator._chat_with_retry.call_args_list[1]
    assert kwargs.get("think") is False


def test_split_segment_by_text_with_words():
    from src.translator import split_segment_by_text

    segment = {
        "start": 0.0,
        "end": 10.0,
        "text": "Hello world this is a test",
        "speaker": "SPEAKER_01",
        "words": [
            {"word": "Hello", "start": 0.0, "end": 2.0},
            {"word": "world", "start": 2.0, "end": 4.0},
            {"word": "this", "start": 4.0, "end": 6.0},
            {"word": "is", "start": 6.0, "end": 7.0},
            {"word": "a", "start": 7.0, "end": 8.0},
            {"word": "test", "start": 8.0, "end": 10.0},
        ],
    }

    split_texts = ["Hello world", "this is a test"]
    sub_segs = split_segment_by_text(segment, split_texts)

    assert len(sub_segs) == 2
    assert sub_segs[0]["text"] == "Hello world"
    assert sub_segs[0]["start"] == 0.0
    assert sub_segs[0]["end"] == 4.0
    assert sub_segs[0]["speaker"] == "SPEAKER_01"

    assert sub_segs[1]["text"] == "this is a test"
    assert sub_segs[1]["start"] == 4.0
    assert sub_segs[1]["end"] == 10.0
    assert sub_segs[1]["speaker"] == "SPEAKER_01"


def test_split_segment_by_text_no_words():
    from src.translator import split_segment_by_text

    segment = {
        "start": 0.0,
        "end": 10.0,
        "text": "Hello world this is a test",
        "speaker": "SPEAKER_01",
    }

    split_texts = ["Hello world", "this is a test"]
    sub_segs = split_segment_by_text(segment, split_texts)

    assert len(sub_segs) == 2
    assert sub_segs[0]["start"] == 0.0
    assert sub_segs[0]["end"] == 5.0
    assert sub_segs[1]["start"] == 5.0
    assert sub_segs[1]["end"] == 10.0


def test_review_diarization_splits():
    from unittest.mock import MagicMock

    from src.translator import Translator

    translator = Translator(ollama_url="http://localhost:11434")

    # Mock response containing a correction and a split
    resp = {
        "message": {
            "content": """
<reasoning>
Segment 0 is actually SPEAKER_02.
Segment 1 should be split between SPEAKER_02 and SPEAKER_01.
</reasoning>
```json
{
  "corrections": [
    {"index": 0, "new_speaker": "SPEAKER_02"}
  ],
  "splits": [
    {
      "index": 1,
      "parts": [
        {"text": "Hello", "speaker": "SPEAKER_02"},
        {"text": "world", "speaker": "SPEAKER_01"}
      ]
    }
  ]
}
```
"""
        }
    }

    translator._chat_with_retry = MagicMock(return_value=resp)

    # We add SPEAKER_02 to segments so SPEAKER_02 is recognized as a valid speaker,
    # preventing the warning and speaker override fallback.
    segments = [
        {"start": 0.0, "end": 2.0, "text": "Hi", "speaker": "SPEAKER_01"},
        {
            "start": 2.0,
            "end": 6.0,
            "text": "Hello world",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "Hello", "start": 2.0, "end": 4.0},
                {"word": "world", "start": 4.0, "end": 6.0},
            ],
        },
        {"start": 6.0, "end": 8.0, "text": "Bye", "speaker": "SPEAKER_02"},
    ]

    reviewed = translator.review_diarization(segments)

    # We expect 4 segments:
    # 1. Corrected segment 0 to SPEAKER_02
    # 2. Segment 1 split into 2 subsegments (one with speaker SPEAKER_02, other SPEAKER_01)
    # 3. Segment 2 unchanged (SPEAKER_02)
    assert len(reviewed) == 4
    assert reviewed[0]["speaker"] == "SPEAKER_02"
    assert reviewed[0]["text"] == "Hi"

    assert reviewed[1]["text"] == "Hello"
    assert reviewed[1]["speaker"] == "SPEAKER_02"
    assert reviewed[1]["start"] == 2.0
    assert reviewed[1]["end"] == 4.0

    assert reviewed[2]["text"] == "world"
    assert reviewed[2]["speaker"] == "SPEAKER_01"
    assert reviewed[2]["start"] == 4.0
    assert reviewed[2]["end"] == 6.0

    assert reviewed[3]["text"] == "Bye"
    assert reviewed[3]["speaker"] == "SPEAKER_02"
