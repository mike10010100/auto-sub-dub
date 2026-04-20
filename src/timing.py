"""Pure-logic timing helpers. Kept model-free so it's cheap to import in tests."""


def annotate_effective_windows(segments, min_duration=1.8, max_steal=1.0, audio_end=None):
    """
    For each segment below `min_duration`, steal silence from the gap on
    either side (up to `max_steal` seconds) to widen the placement window.
    This gives TTS enough room to speak the translation without having to
    be speed-clamped into chipmunk territory. Mutates segments in place.
    """
    n = len(segments)
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        dur = end - start

        prev_end = segments[i - 1]["end"] if i > 0 else 0.0
        next_start = (
            segments[i + 1]["start"]
            if i + 1 < n
            else (audio_end if audio_end is not None else end + max_steal)
        )
        left_gap = max(0.0, start - prev_end)
        right_gap = max(0.0, next_start - end)

        if dur >= min_duration:
            seg["effective_start"] = start
            seg["effective_end"] = end
            continue

        need = min_duration - dur
        left_take = min(need / 2, left_gap, max_steal)
        right_take = min(need - left_take, right_gap, max_steal)
        # If one side was saturated, try to recover from the other.
        if left_take + right_take < need:
            left_take = min(need - right_take, left_gap, max_steal)

        seg["effective_start"] = start - left_take
        seg["effective_end"] = end + right_take
