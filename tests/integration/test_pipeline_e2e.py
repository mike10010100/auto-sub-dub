"""
End-to-end integration test.

Runs the full pipeline (Demucs → WhisperX → Ollama → XTTS → remux) on the
bundled sample video and verifies that the final dubbed artifact exists and
decodes as a valid video with both video and audio streams.

This is NOT part of `mise run is-valid`. It requires:
  - A functional model environment (CUDA/MPS/CPU with enough headroom).
  - A running Ollama instance with the configured models pulled.
  - A Hugging Face token for pyannote diarization.
  - `sample_video.mp4` at the repo root.

Invoke via `mise run integration`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = REPO_ROOT / "sample_video.mp4"


def _probe_streams(path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"]


@pytest.mark.integration
def test_pipeline_produces_dubbed_video():
    if not SAMPLE.exists():
        pytest.skip(f"sample_video.mp4 not present at {SAMPLE}")

    project_dir = REPO_ROOT / "output" / SAMPLE.stem
    if project_dir.exists():
        shutil.rmtree(project_dir)

    from main import main  # import late so import-time side effects are lazy

    main(str(SAMPLE), target_lang="English")

    dubbed = project_dir / f"dubbed_{SAMPLE.name}"
    assert dubbed.exists(), f"expected dubbed video at {dubbed}"
    assert dubbed.stat().st_size > 0

    streams = _probe_streams(dubbed)
    codec_types = {s["codec_type"] for s in streams}
    assert "video" in codec_types, "dubbed output missing video stream"
    assert "audio" in codec_types, "dubbed output missing audio stream"

    assert (project_dir / "transcript.json").exists()
    assert (project_dir / "transcript_translated.json").exists()
    assert (project_dir / "final_audio.wav").exists()
