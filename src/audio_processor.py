import json
import logging
import re
import subprocess
from pathlib import Path

import ffmpeg

logger = logging.getLogger(__name__)

# Map the project's target-language display names to ISO 639-2/B codes
# that FFmpeg / container subtitle streams typically carry.
LANG_TO_ISO3 = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Portuguese": "por",
    "Polish": "pol",
    "Turkish": "tur",
    "Russian": "rus",
    "Dutch": "nld",
    "Czech": "ces",
    "Arabic": "ara",
    "Chinese": "zho",
    "Japanese": "jpn",
    "Korean": "kor",
    "Hungarian": "hun",
    "Hindi": "hin",
}
# Containers sometimes tag streams with 639-2/T or a 2-letter code instead.
LANG_ALIASES = {
    "eng": {"eng", "en"},
    "spa": {"spa", "es"},
    "fra": {"fra", "fre", "fr"},
    "deu": {"deu", "ger", "de"},
    "ita": {"ita", "it"},
    "por": {"por", "pt"},
    "pol": {"pol", "pl"},
    "tur": {"tur", "tr"},
    "rus": {"rus", "ru"},
    "nld": {"nld", "dut", "nl"},
    "ces": {"ces", "cze", "cs"},
    "ara": {"ara", "ar"},
    "zho": {"zho", "chi", "zh"},
    "jpn": {"jpn", "ja"},
    "kor": {"kor", "ko"},
    "hun": {"hun", "hu"},
    "hin": {"hin", "hi"},
}


def _srt_time_to_seconds(ts):
    h, m, rest = ts.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(path):
    """Parse an SRT file into [{'start','end','text'}, ...]."""
    with open(path, encoding="utf-8", errors="replace") as f:
        data = f.read()
    entries = []
    blocks = re.split(r"\n\s*\n", data.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        # First line may be the numeric index; find the timing line.
        timing_idx = 0 if "-->" in lines[0] else 1
        if timing_idx >= len(lines) or "-->" not in lines[timing_idx]:
            continue
        start_s, end_s = (p.strip() for p in lines[timing_idx].split("-->"))
        try:
            start = _srt_time_to_seconds(start_s)
            end = _srt_time_to_seconds(end_s)
        except ValueError:
            continue
        text_lines = lines[timing_idx + 1 :]
        # Strip ASS/SSA style tags ({\i1}), HTML tags, and speaker labels.
        text = " ".join(text_lines)
        text = re.sub(r"\{[^}]*\}", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.strip()
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


class AudioProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path):  # pragma: no cover  (ffmpeg IO)
        """Extract 16kHz mono audio for WhisperX / diarization."""
        logger.info(f"Extracting ASR audio from {video_path}...")
        audio_path = self.temp_dir / "original_audio.wav"
        try:
            (
                ffmpeg.input(video_path)
                .output(str(audio_path), acodec="pcm_s16le", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
            raise

    def extract_audio_hq(self, video_path):  # pragma: no cover  (ffmpeg IO)
        """
        Extract 44.1 kHz stereo audio for XTTS reference cloning.
        XTTS v2 is trained on ~22 kHz+ and degrades noticeably on the 16 kHz
        mono ASR track — so we keep two parallel tracks: one tuned for ASR,
        one for voice cloning.
        """
        logger.info(f"Extracting HQ audio from {video_path}...")
        audio_path = self.temp_dir / "original_audio_hq.wav"
        try:
            (
                ffmpeg.input(video_path)
                .output(str(audio_path), acodec="pcm_s16le", ac=2, ar="44100")
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
            raise

    def separate_vocals(self, audio_path):  # pragma: no cover  (demucs subprocess)
        """Separates vocals from background audio using Demucs."""
        logger.info(f"Separating vocals using Demucs for {audio_path}...")
        # Demucs CLI is often easier to use directly from Python
        # We use the 'htdemucs' model by default
        try:
            subprocess.run(
                ["demucs", "--two-stems", "vocals", "-o", str(self.temp_dir), str(audio_path)],
                check=True,
                capture_output=True,
                text=True,
            )

            # Demucs creates a folder structure: output/temp/htdemucs/original_audio/vocals.wav
            # and output/temp/htdemucs/original_audio/no_vocals.wav
            base_name = Path(audio_path).stem
            model_name = "htdemucs"

            vocals_path = self.temp_dir / model_name / base_name / "vocals.wav"
            background_path = self.temp_dir / model_name / base_name / "no_vocals.wav"

            return vocals_path, background_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs separation failed: {e.stderr if e.stderr else str(e)}")
            raise RuntimeError(f"Demucs failed: {e.stderr if e.stderr else str(e)}") from e
        except FileNotFoundError as e:
            logger.error(f"Demucs executable not found. Is it installed and in PATH? {e}")
            raise RuntimeError("Demucs executable not found. Please install demucs.") from e

    def extract_target_subtitles(
        self, video_path, target_lang
    ):  # pragma: no cover  (ffprobe/ffmpeg IO)
        """
        If the source video carries a subtitle stream in `target_lang`,
        extract it to SRT in temp_dir. Used as a reconciliation hint for
        translation, not as gospel (dubtitles are often condensed).

        Returns the SRT path if a matching stream was found and extracted,
        otherwise None.
        """
        iso = LANG_TO_ISO3.get(target_lang)
        if not iso:
            logger.info(f"No ISO mapping for '{target_lang}'; skipping sub extraction.")
            return None
        accepted = LANG_ALIASES.get(iso, {iso})

        try:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_streams",
                    "-select_streams",
                    "s",
                    str(video_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"ffprobe failed, skipping subtitle reconciliation: {e}")
            return None

        streams = json.loads(probe.stdout or "{}").get("streams", [])
        match_idx = None
        match_codec = None
        for s in streams:
            tag = (s.get("tags") or {}).get("language", "").lower()
            if tag in accepted:
                match_idx = s.get("index")
                match_codec = s.get("codec_name")
                break

        if match_idx is None:
            logger.info(f"No '{target_lang}' ({iso}) subtitle stream found in source.")
            return None

        out_path = self.temp_dir / f"subtitles_{iso}.srt"
        logger.info(
            f"Extracting {target_lang} subtitle stream (index={match_idx}, codec={match_codec}) → {out_path}"
        )
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    "-map",
                    f"0:{match_idx}",
                    "-c:s",
                    "srt",
                    str(out_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.warning(f"Subtitle extraction failed: {e.stderr.decode() if e.stderr else e}")
            return None
        return out_path

    def noisegate(self, segment, threshold_db=-40):
        """Simple noisegate to zero out samples below a threshold."""
        import numpy as np
        from pydub import AudioSegment

        # Convert to numpy for fast processing
        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(segment.sample_width, np.int16)
        peak = float(2 ** (8 * segment.sample_width - 1))

        samples = np.frombuffer(segment.raw_data, dtype=dtype).astype(np.float32)

        # Simple windowed RMS gate
        win_size = int(segment.frame_rate * 0.02)  # 20ms windows
        n_full = (len(samples) // win_size) * win_size
        if n_full == 0:
            return segment

        frames = samples[:n_full].reshape(-1, win_size)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        rms_db = 20 * np.log10(rms / peak + 1e-12)

        # Zero out frames below threshold
        gate = np.where(rms_db > threshold_db, 1.0, 0.0).astype(np.float32)

        # Smooth gate transitions slightly to prevent clicks
        gate_smoothed = np.repeat(gate, win_size)

        # Align lengths
        if len(gate_smoothed) < len(samples):
            gate_smoothed = np.concatenate(
                [gate_smoothed, np.zeros(len(samples) - len(gate_smoothed))]
            )

        samples_gated = samples * gate_smoothed[: len(samples)]

        return AudioSegment(
            data=samples_gated.astype(dtype).tobytes(),
            sample_width=segment.sample_width,
            frame_rate=segment.frame_rate,
            channels=segment.channels,
        )

    def focus_vocals(self, audio_path):  # pragma: no cover
        """
        Preprocessing step for diarization: apply a bandpass filter (human speech range)
        and a noisegate to remove hiss/reverb that confuses the diarizer.
        """
        from pydub import AudioSegment

        logger.info(f"Applying Vocal Focus filter to {audio_path}...")
        audio = AudioSegment.from_wav(audio_path)

        # Bandpass: Keep 100Hz to 8000Hz (standard human speech range)
        # We use high_pass and low_pass sequentially for maximum compatibility
        focused = audio.high_pass_filter(100).low_pass_filter(8000)

        # Apply noisegate to remove residual isolation artifacts/reverb
        focused = self.noisegate(focused, threshold_db=-35)

        output_path = Path(str(audio_path).replace(".wav", "_focused.wav"))
        focused.export(output_path, format="wav")
        return output_path

    def match_loudness(
        self, target_segment, reference_path, max_gain_db=12.0
    ):  # pragma: no cover  (pyloudnorm + audio IO)
        """
        Normalize `target_segment` (AudioSegment, the dubbed track) to match
        the integrated loudness of `reference_path` (WAV, the original
        isolated vocals) per EBU R128 / ITU-R BS.1770. Keeps the dub sitting
        at the same subjective level as the source voice so ducking works on
        matched material instead of fighting a too-quiet or too-loud dub.
        """
        import numpy as np
        import pyloudnorm as pyln
        from pydub import AudioSegment

        def _seg_to_float(seg):
            dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
            dt = dtype_map.get(seg.sample_width, np.int16)
            pk = float(2 ** (8 * seg.sample_width - 1))
            arr = np.frombuffer(seg.raw_data, dtype=dt).astype(np.float32) / pk
            if seg.channels == 2:
                arr = arr.reshape(-1, 2)
            return arr

        ref_seg = AudioSegment.from_file(str(reference_path))
        ref_arr = _seg_to_float(ref_seg)
        ref_lufs = pyln.Meter(ref_seg.frame_rate).integrated_loudness(ref_arr)

        tgt_arr = _seg_to_float(target_segment)
        tgt_lufs = pyln.Meter(target_segment.frame_rate).integrated_loudness(tgt_arr)

        if not np.isfinite(tgt_lufs) or not np.isfinite(ref_lufs):
            logger.info(f"Loudness match skipped (ref={ref_lufs}, tgt={tgt_lufs}).")
            return target_segment

        gain_db = float(np.clip(ref_lufs - tgt_lufs, -max_gain_db, max_gain_db))
        logger.info(
            f"Loudness match: ref={ref_lufs:.1f} LUFS, dub={tgt_lufs:.1f} LUFS, "
            f"applying {gain_db:+.1f} dB"
        )
        return target_segment.apply_gain(gain_db)

    def duck_audio(
        self,
        vocals,
        background,
        duck_db=-15,  # pragma: no cover  (audio IO)
        threshold_db=-40,
        attack_ms=50,
        release_ms=400,
        frame_ms=10,
    ):
        """
        Sidechain-style ducking with smoothed envelope.

        Compute an RMS envelope of `vocals` at `frame_ms` resolution, gate it
        at `threshold_db` to produce a target gain curve (0 dB quiet / duck_db
        loud), smooth with a one-pole IIR (fast attack, slow release) to
        eliminate the pumping that hard-edged block ducking produces, then
        apply the per-sample gain to `background` and overlay vocals.
        """
        import numpy as np
        from pydub import AudioSegment

        logger.info(
            f"Ducking background by {duck_db}dB "
            f"(attack {attack_ms}ms, release {release_ms}ms, frame {frame_ms}ms)"
        )

        sr = background.frame_rate
        channels = background.channels
        sample_width = background.sample_width

        # Align lengths and formats.
        max_len_ms = max(len(vocals), len(background))
        if len(vocals) < max_len_ms:
            vocals = vocals + AudioSegment.silent(
                duration=max_len_ms - len(vocals), frame_rate=vocals.frame_rate
            )
        if len(background) < max_len_ms:
            background = background + AudioSegment.silent(
                duration=max_len_ms - len(background), frame_rate=sr
            )
        vocals_aligned = vocals.set_frame_rate(sr).set_channels(channels)

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sample_width, np.int16)
        peak = float(2 ** (8 * sample_width - 1))

        bg = np.frombuffer(background.raw_data, dtype=dtype).astype(np.float32)
        v = np.frombuffer(vocals_aligned.raw_data, dtype=dtype).astype(np.float32)

        if channels == 2:
            bg_stereo = bg.reshape(-1, 2)
            v_mono = v.reshape(-1, 2).mean(axis=1)
        else:
            bg_stereo = bg
            v_mono = v

        win = max(1, int(sr * frame_ms / 1000))
        n_full = (len(v_mono) // win) * win
        if n_full == 0:
            return vocals_aligned.overlay(background)
        frames = v_mono[:n_full].reshape(-1, win)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
        rms_db = 20 * np.log10(rms / peak + 1e-12)

        target_db = np.where(rms_db > threshold_db, float(duck_db), 0.0).astype(np.float32)

        # One-pole smoothing with asymmetric time constants.
        a_att = 1.0 - float(np.exp(-frame_ms / max(1e-6, attack_ms)))
        a_rel = 1.0 - float(np.exp(-frame_ms / max(1e-6, release_ms)))
        smoothed = np.empty_like(target_db)
        g = 0.0
        for i, t in enumerate(target_db):
            # "attack" = envelope getting MORE negative (ducking harder).
            coef = a_att if t < g else a_rel
            g += coef * (t - g)
            smoothed[i] = g

        gain_lin = np.power(10.0, smoothed / 20.0)
        per_sample_gain = np.repeat(gain_lin, win)
        # Align to the background's actual sample count (resampling can drift
        # by a few samples vs. the vocal-mono length).
        target_len = bg_stereo.shape[0]
        if len(per_sample_gain) < target_len:
            tail = per_sample_gain[-1] if len(per_sample_gain) else 1.0
            per_sample_gain = np.concatenate(
                [
                    per_sample_gain,
                    np.full(target_len - len(per_sample_gain), tail, dtype=np.float32),
                ]
            )
        per_sample_gain = per_sample_gain[:target_len]

        if channels == 2:
            ducked = bg_stereo * per_sample_gain[:, None]
        else:
            ducked = bg_stereo * per_sample_gain

        np.clip(ducked, -peak, peak - 1, out=ducked)
        ducked_bytes = ducked.astype(dtype).tobytes()

        ducked_seg = AudioSegment(
            data=ducked_bytes,
            sample_width=sample_width,
            frame_rate=sr,
            channels=channels,
        )
        return vocals_aligned.overlay(ducked_seg)


if __name__ == "__main__":
    # Test stub
    import sys

    if len(sys.argv) > 1:
        proc = AudioProcessor()
        orig_audio = proc.extract_audio(sys.argv[1])
        v, b = proc.separate_vocals(orig_audio)
        print(f"Vocals: {v}")
        print(f"Background: {b}")
