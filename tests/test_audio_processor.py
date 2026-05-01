import textwrap

import pytest

from src.audio_processor import (
    LANG_ALIASES,
    LANG_TO_ISO3,
    AudioProcessor,
    _srt_time_to_seconds,
    parse_srt,
)


def test_audio_processor_init_creates_dirs(tmp_path):
    ap = AudioProcessor(output_dir=tmp_path / "out")
    assert ap.output_dir.is_dir()
    assert ap.temp_dir.is_dir()
    assert ap.temp_dir.parent == ap.output_dir


def test_parse_srt_skips_one_line_blocks(tmp_path):
    srt = tmp_path / "f.srt"
    srt.write_text("JustOneLine\n")
    assert parse_srt(srt) == []


def test_srt_time_to_seconds_basic():
    assert _srt_time_to_seconds("00:00:01,500") == pytest.approx(1.5)
    assert _srt_time_to_seconds("01:02:03,004") == pytest.approx(3723.004)


def test_srt_time_to_seconds_zero():
    assert _srt_time_to_seconds("00:00:00,000") == 0.0


def test_parse_srt_strips_html_and_ass_tags(tmp_path):
    srt = tmp_path / "a.srt"
    srt.write_text(
        textwrap.dedent("""
        1
        00:00:01,000 --> 00:00:02,500
        Hello, world.

        2
        00:00:03,100 --> 00:00:05,800
        <i>This is a test.</i>
        Second line.

        3
        00:01:00,000 --> 00:01:02,000
        {\\an8}Styled caption
    """).strip()
    )
    entries = parse_srt(srt)
    assert len(entries) == 3
    assert entries[0] == {"start": 1.0, "end": 2.5, "text": "Hello, world."}
    assert "<i>" not in entries[1]["text"]
    assert "This is a test." in entries[1]["text"]
    assert "Second line." in entries[1]["text"]
    assert entries[2]["text"] == "Styled caption"


def test_parse_srt_handles_no_index_line(tmp_path):
    # Some SRTs omit the numeric index; the timing line is first.
    srt = tmp_path / "b.srt"
    srt.write_text("00:00:01,000 --> 00:00:02,000\nNo index here.\n")
    entries = parse_srt(srt)
    assert entries == [{"start": 1.0, "end": 2.0, "text": "No index here."}]


def test_parse_srt_skips_malformed_blocks(tmp_path):
    srt = tmp_path / "c.srt"
    srt.write_text(
        textwrap.dedent("""
        1
        not-a-timing-line
        Broken block

        2
        00:00:05,000 --> 00:00:06,000
        Good one
    """).strip()
    )
    entries = parse_srt(srt)
    assert len(entries) == 1
    assert entries[0]["text"] == "Good one"


def test_parse_srt_skips_empty_text(tmp_path):
    srt = tmp_path / "d.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:02,000\n{\\pos(0,0)}\n")
    entries = parse_srt(srt)
    assert entries == []


def test_parse_srt_skips_bad_timestamps(tmp_path):
    srt = tmp_path / "e.srt"
    srt.write_text("1\nBAD --> ALSOBAD\nLine\n")
    entries = parse_srt(srt)
    assert entries == []


def test_noisegate_zeros_silent_audio():
    import numpy as np
    from pydub import AudioSegment

    ap = AudioProcessor()
    # Create 1 second of silence (-inf dB)
    silence = (
        AudioSegment.silent(duration=1000, frame_rate=44100).set_channels(1).set_sample_width(2)
    )
    gated = ap.noisegate(silence, threshold_db=-40)

    # Gated silence should still be silence
    assert gated.dBFS == float("-inf")

    # Create audio with a loud part and a quiet part
    # 0.5s of -10dB sine wave, 0.5s of -60dB silence
    duration = 1000
    fs = 44100
    t = np.linspace(0, duration / 1000, fs, endpoint=False)

    # Loud part (-3dBFS approx)
    loud_samples = (np.sin(2 * np.pi * 440 * t[: fs // 2]) * 32767 * 0.7).astype(np.int16)
    # Quiet part (-100dBFS)
    quiet_samples = np.zeros(fs - fs // 2, dtype=np.int16)

    combined_samples = np.concatenate([loud_samples, quiet_samples])
    audio = AudioSegment(combined_samples.tobytes(), frame_rate=fs, sample_width=2, channels=1)

    # Apply gate at -40dB
    gated_audio = ap.noisegate(audio, threshold_db=-40)

    # Verify samples: first half should be preserved, second half should be zeroed
    gated_samples = np.frombuffer(gated_audio.raw_data, dtype=np.int16)

    # First 100ms should be roughly equal (minus windowing/smoothing effects)
    assert np.max(np.abs(gated_samples[: fs // 4])) > 1000
    # Last 100ms should be exactly zero
    assert np.all(gated_samples[-fs // 4 :] == 0)


def test_lang_aliases_contain_primary_iso():
    for iso, aliases in LANG_ALIASES.items():
        assert iso in aliases, f"{iso} should include itself in its alias set"


def test_lang_to_iso3_has_aliases():
    for iso in LANG_TO_ISO3.values():
        assert iso in LANG_ALIASES, f"missing alias entry for {iso}"
