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
