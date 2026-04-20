import pytest

from src.timing import annotate_effective_windows


def _mk(start, end):
    return {"start": start, "end": end}


def _approx(v):
    return pytest.approx(v, rel=1e-6)


def test_annotate_no_widening_when_duration_ok():
    segs = [_mk(0.0, 2.0), _mk(3.0, 6.0)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=10.0)
    assert segs[0]["effective_start"] == 0.0
    assert segs[0]["effective_end"] == 2.0
    assert segs[1]["effective_start"] == 3.0
    assert segs[1]["effective_end"] == 6.0


def test_annotate_symmetric_steal_from_both_sides():
    # 1.0s segment, 2.0s gaps on either side, min_duration=1.8 → need 0.8s,
    # takes 0.4s from each side.
    segs = [_mk(0.0, 1.0), _mk(3.0, 4.0), _mk(6.0, 7.0)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=10.0)
    mid = segs[1]
    assert mid["effective_start"] == _approx(3.0 - 0.4)
    assert mid["effective_end"] == _approx(4.0 + 0.4)


def test_annotate_asymmetric_when_one_side_saturated():
    # Left gap is tiny (0.1s), right gap is huge — we still hit the 1.8s target
    # by stealing more from the right.
    segs = [_mk(0.0, 1.9), _mk(2.0, 2.5), _mk(8.0, 9.0)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=10.0)
    mid = segs[1]
    # Window needs 1.3s extension total; left has only 0.1s available.
    assert mid["effective_start"] == _approx(1.9)
    assert mid["effective_end"] - mid["effective_start"] >= 1.3


def test_annotate_respects_max_steal_per_side():
    # Each side has plenty of gap, but max_steal caps the per-side grab.
    segs = [_mk(0.0, 0.1), _mk(5.0, 5.1), _mk(20.0, 20.1)]
    annotate_effective_windows(segs, min_duration=5.0, max_steal=1.0, audio_end=30.0)
    mid = segs[1]
    assert mid["effective_start"] >= 4.0  # capped at 1.0s steal
    assert mid["effective_end"] <= 6.1


def test_annotate_first_segment_left_gap_is_from_zero():
    segs = [_mk(0.5, 1.0), _mk(5.0, 6.0)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=10.0)
    # First segment treats prev_end=0, so has 0.5s of left room available.
    assert segs[0]["effective_start"] >= 0.0
    assert segs[0]["effective_end"] > 1.0


def test_annotate_last_segment_right_uses_audio_end():
    segs = [_mk(0.0, 2.0), _mk(3.0, 3.5)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=5.0)
    last = segs[1]
    assert last["effective_end"] <= 5.0


def test_annotate_last_segment_without_audio_end_extends():
    segs = [_mk(0.0, 0.5)]
    annotate_effective_windows(segs, min_duration=1.8, max_steal=1.0, audio_end=None)
    # With audio_end=None, the helper assumes end+max_steal of silence available.
    only = segs[0]
    assert only["effective_end"] > 0.5
