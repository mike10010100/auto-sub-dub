import os
import subprocess
import ffmpeg
import demucs.separate
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio(self, video_path):
        """Extracts the audio track from a video file."""
        logger.info(f"Extracting audio from {video_path}...")
        audio_path = self.temp_dir / "original_audio.wav"
        try:
            (
                ffmpeg
                .input(video_path)
                .output(str(audio_path), acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")
            raise

    def separate_vocals(self, audio_path):
        """Separates vocals from background audio using Demucs."""
        logger.info(f"Separating vocals using Demucs for {audio_path}...")
        # Demucs CLI is often easier to use directly from Python
        # We use the 'htdemucs' model by default
        try:
            subprocess.run([
                "demucs",
                "--two-stems", "vocals",
                "-o", str(self.temp_dir),
                str(audio_path)
            ], check=True)
            
            # Demucs creates a folder structure: output/temp/htdemucs/original_audio/vocals.wav
            # and output/temp/htdemucs/original_audio/no_vocals.wav
            base_name = Path(audio_path).stem
            model_name = "htdemucs"
            
            vocals_path = self.temp_dir / model_name / base_name / "vocals.wav"
            background_path = self.temp_dir / model_name / base_name / "no_vocals.wav"
            
            return vocals_path, background_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs separation failed: {e}")
            raise

    def duck_audio(self, vocals, background, duck_db=-15,
                   threshold_db=-40, attack_ms=50, release_ms=400,
                   frame_ms=10):
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
        target_len = bg_stereo.shape[0] if channels == 2 else bg_stereo.shape[0]
        if len(per_sample_gain) < target_len:
            tail = per_sample_gain[-1] if len(per_sample_gain) else 1.0
            per_sample_gain = np.concatenate([
                per_sample_gain,
                np.full(target_len - len(per_sample_gain), tail, dtype=np.float32),
            ])
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
