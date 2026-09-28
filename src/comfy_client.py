import logging
import os
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def parse_whisperx_srt(srt_text: str) -> list[dict[str, Any]]:
    """Parse WhisperX SRT output text containing speaker labels into segment dictionaries."""
    import re

    from src.audio_processor import _srt_time_to_seconds

    blocks = re.split(r"\n\s*\n", srt_text.strip())
    segments = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        timing_idx = 0 if "-->" in lines[0] else 1
        if timing_idx >= len(lines) or "-->" not in lines[timing_idx]:
            continue
        start_s, end_s = (p.strip() for p in lines[timing_idx].split("-->"))
        try:
            start = _srt_time_to_seconds(start_s)
            end = _srt_time_to_seconds(end_s)
        except ValueError:
            continue
        raw_text = " ".join(lines[timing_idx + 1 :])
        m = re.match(r"^(\d+)\s*(.*)$", raw_text)
        if m:
            spk_num = int(m.group(1))
            spk = f"SPEAKER_{spk_num:02d}"
            text = m.group(2).strip()
        else:
            spk = "SPEAKER_00"
            text = raw_text.strip()
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": spk,
                "text": text,
            }
        )
    return segments


class ComfyClient:
    """Client for interacting with a remote or local ComfyUI instance."""

    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("COMFYUI_URL", "http://localhost:8188")).rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:  # pragma: no cover
        """Check if ComfyUI is responding."""
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def upload_file(
        self, file_path: str | Path, subfolder: str = "", overwrite: bool = True
    ) -> str:  # pragma: no cover
        """Upload an audio/video file to ComfyUI's input directory."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found for upload: {path}")

        temp_flac = None
        upload_path = path
        try:
            # Compress large WAV files (> 80MB) to lossless FLAC to avoid 413 request limits
            if path.suffix.lower() == ".wav" and path.stat().st_size > 80 * 1024 * 1024:
                from pydub import AudioSegment

                temp_flac = path.with_suffix(".upload_tmp.flac")
                logger.info(
                    f"Compressing large audio {path.name} ({path.stat().st_size // (1024 * 1024)}MB) "
                    f"to lossless FLAC for ComfyUI upload..."
                )
                seg = AudioSegment.from_file(path)
                seg.export(temp_flac, format="flac")
                upload_path = temp_flac

            url = f"{self.base_url}/upload/image"
            with open(upload_path, "rb") as f:
                files = {"image": (upload_path.name, f)}
                data = {"overwrite": "true" if overwrite else "false", "subfolder": subfolder}
                resp = requests.post(url, files=files, data=data, timeout=self.timeout)

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to upload {upload_path.name}: {resp.status_code} {resp.text}"
                )

            res_json = resp.json()
            return res_json.get("name", upload_path.name)
        finally:
            if temp_flac and temp_flac.exists():
                temp_flac.unlink(missing_ok=True)

    def download_file(
        self,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
        output_path: str | Path | None = None,
    ) -> Path:  # pragma: no cover
        """Download a generated artifact from ComfyUI."""
        url = f"{self.base_url}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        resp = requests.get(url, params=params, stream=True, timeout=self.timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download {filename}: {resp.status_code} {resp.text}")

        target = Path(output_path) if output_path else Path(filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return target

    def queue_prompt(self, prompt: dict[str, Any]) -> str:  # pragma: no cover
        """Submit a prompt graph to ComfyUI and return the prompt ID."""
        url = f"{self.base_url}/prompt"
        resp = requests.post(url, json={"prompt": prompt}, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to queue prompt: {resp.status_code} {resp.text}")

        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI validation error: {data['error']}")
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"No prompt_id in response: {data}")
        return prompt_id

    def wait_for_prompt(
        self,
        prompt_id: str,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:  # pragma: no cover
        """Poll ComfyUI history until prompt execution completes or fails."""
        start_time = time.time()
        url = f"{self.base_url}/history/{prompt_id}"

        while time.time() - start_time < timeout:
            try:
                resp = requests.get(url, timeout=10.0)
                if resp.status_code == 200:
                    data = resp.json()
                    if prompt_id in data:
                        item = data[prompt_id]
                        status = item.get("status", {})
                        if status.get("status_str") == "error":
                            messages = status.get("messages", [])
                            err_msg = str(messages)
                            for msg in messages:
                                if (
                                    len(msg) > 1
                                    and isinstance(msg[1], dict)
                                    and "exception_message" in msg[1]
                                ):
                                    err_msg = msg[1]["exception_message"]
                                    break
                            raise RuntimeError(f"ComfyUI execution failed: {err_msg}")
                        if status.get("completed", False):
                            return item.get("outputs", {})
            except requests.RequestException as e:
                logger.warning(f"Error checking prompt status: {e}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Prompt {prompt_id} timed out after {timeout} seconds")

    # --- Pure Workflow Builders (Tested by Unit Suite) ---

    def build_separator_workflow(
        self,
        audio_filename: str,
        model_filename: str = "mel_band_roformer_vocals_kim_ft.ckpt",
        window_size: int = 512,
        aggression: int = 10,
        batch_size: int = 4,
        normalization: float = 0.9,
    ) -> dict[str, Any]:
        """Build workflow graph for Mel-Band Roformer vocal separation."""
        return {
            "1": {
                "inputs": {"audio": audio_filename},
                "class_type": "LoadAudio",
            },
            "2": {
                "inputs": {
                    "audio": ["1", 0],
                    "model_filename": model_filename,
                    "window_size": window_size,
                    "aggression": aggression,
                    "batch_size": batch_size,
                    "normalization": normalization,
                },
                "class_type": "AudioSeparatorNode",
            },
            "3": {
                "inputs": {
                    "audio": ["2", 0],
                    "filename_prefix": "audio/roformer_vocals",
                },
                "class_type": "SaveAudio",
            },
            "4": {
                "inputs": {
                    "audio": ["2", 1],
                    "filename_prefix": "audio/roformer_background",
                },
                "class_type": "SaveAudio",
            },
        }

    def build_demucs_workflow(
        self,
        audio_filename: str,
        model: str = "💾 Hybrid Transformer",
        shifts: int = 0,
        overlap: float = 0.25,
        target_device: str = "cuda",
    ) -> dict[str, Any]:
        """Build workflow graph for CUDA Demucs vocal separation via ComfyUI."""
        return {
            "1": {
                "inputs": {"audio": audio_filename},
                "class_type": "LoadAudio",
            },
            "2": {
                "inputs": {
                    "input_sound": ["1", 0],
                    "model": model,
                    "shifts": shifts,
                    "overlap": overlap,
                    "custom_segment": False,
                    "segment": 44,
                    "target_device": target_device,
                },
                "class_type": "AudioSeparateDemucs",
            },
            "3": {
                "inputs": {
                    "audio": ["2", 0],
                    "filename_prefix": "audio/demucs_vocals",
                },
                "class_type": "SaveAudio",
            },
            "4": {
                "inputs": {
                    "audio1": ["2", 1],
                    "audio2": ["2", 2],
                    "merge_method": "add",
                },
                "class_type": "AudioMerge",
            },
            "5": {
                "inputs": {
                    "audio1": ["4", 0],
                    "audio2": ["2", 3],
                    "merge_method": "add",
                },
                "class_type": "AudioMerge",
            },
            "6": {
                "inputs": {
                    "audio": ["5", 0],
                    "filename_prefix": "audio/demucs_background",
                },
                "class_type": "SaveAudio",
            },
        }

    def build_whisperx_workflow(
        self,
        audio_filename: str,
        model_type: str = "large-v3",
        batch_size: int = 8,
        if_mutiple_speaker: bool = True,
        use_auth_token: str | None = None,
    ) -> dict[str, Any]:
        """Build workflow graph for WhisperX transcription & diarization via ComfyUI."""
        token = use_auth_token if use_auth_token is not None else os.getenv("HF_TOKEN", "")
        return {
            "1": {
                "inputs": {"audio": audio_filename},
                "class_type": "LoadAudioVideoPath",
            },
            "2": {
                "inputs": {
                    "audio": ["1", 0],
                    "model_type": model_type,
                    "batch_size": batch_size,
                    "if_mutiple_speaker": if_mutiple_speaker,
                    "use_auth_token": token,
                    "if_translate": False,
                    "translator": "alibaba",
                    "to_language": "en",
                },
                "class_type": "WhisperX",
            },
            "3": {
                "inputs": {"srt": ["2", 0]},
                "class_type": "PreViewSRT",
            },
        }

    def build_f5_tts_workflow(
        self,
        sample_audio_filename: str,
        sample_text: str,
        speech_text: str,
        speed: float = 1.0,
        seed: int = 42,
        model: str = "F5",
        model_type: str = "F5TTS_Base",
    ) -> dict[str, Any]:
        """Build workflow graph for F5-TTS speech synthesis."""
        return {
            "1": {
                "inputs": {"audio": sample_audio_filename},
                "class_type": "LoadAudio",
            },
            "2": {
                "inputs": {
                    "sample_audio": ["1", 0],
                    "sample_text": sample_text or "Dialogue snippet",
                    "speech": speech_text,
                    "seed": seed,
                    "model": model,
                    "vocoder": "auto",
                    "speed": speed,
                    "model_type": model_type,
                },
                "class_type": "F5TTSAudioInputs",
            },
            "3": {
                "inputs": {
                    "audio": ["2", 0],
                    "filename_prefix": "audio/f5tts_dub",
                },
                "class_type": "SaveAudio",
            },
        }

    def build_latentsync_workflow(
        self,
        video_filename: str,
        audio_filename: str,
        seed: int = 1247,
        inference_steps: int = 20,
        lips_expression: float = 1.5,
    ) -> dict[str, Any]:
        """Build workflow graph for LatentSync lip synchronization."""
        return {
            "1": {
                "inputs": {"file": video_filename},
                "class_type": "LoadVideo",
            },
            "2": {
                "inputs": {"video": ["1", 0]},
                "class_type": "GetVideoComponents",
            },
            "3": {
                "inputs": {"audio": audio_filename},
                "class_type": "LoadAudio",
            },
            "4": {
                "inputs": {
                    "images": ["2", 0],
                    "audio": ["3", 0],
                    "seed": seed,
                    "lips_expression": lips_expression,
                    "inference_steps": inference_steps,
                },
                "class_type": "LatentSyncNode",
            },
            "5": {
                "inputs": {
                    "images": ["4", 0],
                    "audio": ["4", 1],
                    "fps": ["2", 2],
                },
                "class_type": "CreateVideo",
            },
            "6": {
                "inputs": {
                    "video": ["5", 0],
                    "filename_prefix": "video/latentsync_dub",
                    "format": "auto",
                },
                "class_type": "SaveVideo",
            },
        }

    def build_cosyvoice3_vc_workflow(
        self,
        source_audio_filename: str,
        target_audio_filename: str,
        model_version: str = "Fun-CosyVoice3-0.5B",
        speed: float = 1.0,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Build workflow graph for CosyVoice 3 Voice Conversion timbre transfer."""
        return {
            "1": {
                "inputs": {
                    "model_version": model_version,
                    "download_source": "HuggingFace",
                    "device": "auto",
                },
                "class_type": "FL_CosyVoice3_ModelLoader",
            },
            "2": {
                "inputs": {"audio": source_audio_filename},
                "class_type": "LoadAudio",
            },
            "3": {
                "inputs": {"audio": target_audio_filename},
                "class_type": "LoadAudio",
            },
            "4": {
                "inputs": {
                    "model": ["1", 0],
                    "source_audio": ["2", 0],
                    "target_audio": ["3", 0],
                    "speed": speed,
                    "seed": seed,
                },
                "class_type": "FL_CosyVoice3_VoiceConversion",
            },
            "5": {
                "inputs": {
                    "audio": ["4", 0],
                    "filename_prefix": "audio/cosyvoice3_vc",
                },
                "class_type": "SaveAudio",
            },
        }

    # --- High-Level Operations ---

    def separate_audio(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        model_filename: str = "mel_band_roformer_vocals_kim_ft.ckpt",
    ) -> tuple[Path, Path]:  # pragma: no cover
        """Run Mel-Band Roformer separation via ComfyUI and return (vocals_path, background_path)."""
        logger.info(f"Uploading {audio_path} to ComfyUI for Roformer vocal separation...")
        uploaded_name = self.upload_file(audio_path)

        workflow = self.build_separator_workflow(uploaded_name, model_filename=model_filename)
        prompt_id = self.queue_prompt(workflow)
        logger.info(f"Roformer separation job queued (Prompt ID: {prompt_id}), waiting...")

        outputs = self.wait_for_prompt(prompt_id, timeout=300.0)

        vocals_info = outputs["3"]["audio"][0]
        bg_info = outputs["4"]["audio"][0]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_path = out_dir / "vocals.flac"
        background_path = out_dir / "no_vocals.flac"

        self.download_file(
            vocals_info["filename"],
            subfolder=vocals_info.get("subfolder", ""),
            folder_type=vocals_info.get("type", "output"),
            output_path=vocals_path,
        )
        self.download_file(
            bg_info["filename"],
            subfolder=bg_info.get("subfolder", ""),
            folder_type=bg_info.get("type", "output"),
            output_path=background_path,
        )

        return vocals_path, background_path

    def separate_audio_demucs(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        model: str = "💾 Hybrid Transformer",
    ) -> tuple[Path, Path]:  # pragma: no cover
        """Run CUDA Demucs separation via ComfyUI and return (vocals_path, background_path)."""
        logger.info(f"Uploading {audio_path} to ComfyUI for Demucs vocal separation...")
        uploaded_name = self.upload_file(audio_path)

        workflow = self.build_demucs_workflow(uploaded_name, model=model)
        prompt_id = self.queue_prompt(workflow)
        logger.info(f"Demucs separation job queued (Prompt ID: {prompt_id}), waiting...")

        outputs = self.wait_for_prompt(prompt_id, timeout=600.0)

        vocals_info = outputs["3"]["audio"][0]
        bg_info = outputs["6"]["audio"][0]

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        vocals_path = out_dir / "vocals.flac"
        background_path = out_dir / "no_vocals.flac"

        self.download_file(
            vocals_info["filename"],
            subfolder=vocals_info.get("subfolder", ""),
            folder_type=vocals_info.get("type", "output"),
            output_path=vocals_path,
        )
        self.download_file(
            bg_info["filename"],
            subfolder=bg_info.get("subfolder", ""),
            folder_type=bg_info.get("type", "output"),
            output_path=background_path,
        )

        return vocals_path, background_path

    def transcribe_whisperx(
        self,
        audio_path: str | Path,
        model_type: str = "large-v3",
        batch_size: int = 8,
        if_mutiple_speaker: bool = True,
        auth_token: str | None = None,
    ) -> dict[str, Any]:  # pragma: no cover
        """Run WhisperX transcription and diarization via ComfyUI."""
        logger.info(f"Uploading {audio_path} to ComfyUI for WhisperX transcription...")
        uploaded_name = self.upload_file(audio_path)
        token = auth_token or os.getenv("HF_TOKEN", "")

        workflow = self.build_whisperx_workflow(
            uploaded_name,
            model_type=model_type,
            batch_size=batch_size,
            if_mutiple_speaker=if_mutiple_speaker,
            use_auth_token=token,
        )
        prompt_id = self.queue_prompt(workflow)
        logger.info(f"WhisperX transcription job queued (Prompt ID: {prompt_id}), waiting...")

        outputs = self.wait_for_prompt(prompt_id, timeout=900.0)

        srt_node = outputs.get("3", {})
        srt_data = srt_node.get("srt", [])
        if not srt_data:
            raise RuntimeError(f"WhisperX did not return expected SRT outputs: {outputs}")

        srt_text = srt_data[0]
        segments = parse_whisperx_srt(srt_text)

        # Detect language heuristic or default
        is_japanese = any(ord(c) > 0x3000 for s in segments for c in s.get("text", ""))
        detected_lang = "ja" if is_japanese else "en"

        return {
            "segments": segments,
            "language": detected_lang,
        }

    def synthesize_speech_f5(
        self,
        sample_audio_path: str | Path,
        sample_text: str,
        speech_text: str,
        output_path: str | Path,
        speed: float = 1.0,
        seed: int = 42,
    ) -> Path:  # pragma: no cover
        """Synthesize speech using F5-TTS on ComfyUI."""
        uploaded_sample = self.upload_file(sample_audio_path)
        workflow = self.build_f5_tts_workflow(
            sample_audio_filename=uploaded_sample,
            sample_text=sample_text,
            speech_text=speech_text,
            speed=speed,
            seed=seed,
        )
        prompt_id = self.queue_prompt(workflow)
        outputs = self.wait_for_prompt(prompt_id, timeout=120.0)

        audio_info = outputs["3"]["audio"][0]
        return self.download_file(
            audio_info["filename"],
            subfolder=audio_info.get("subfolder", ""),
            folder_type=audio_info.get("type", "output"),
            output_path=output_path,
        )

    def lip_sync_video(
        self,
        video_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
        seed: int = 1247,
        inference_steps: int = 20,
        lips_expression: float = 1.5,
    ) -> Path:  # pragma: no cover
        """Run LatentSync lip synchronization on ComfyUI and download the result."""
        logger.info(
            f"Uploading video {video_path} and audio {audio_path} to ComfyUI for LatentSync..."
        )
        uploaded_video = self.upload_file(video_path)
        uploaded_audio = self.upload_file(audio_path)

        workflow = self.build_latentsync_workflow(
            video_filename=uploaded_video,
            audio_filename=uploaded_audio,
            seed=seed,
            inference_steps=inference_steps,
            lips_expression=lips_expression,
        )
        prompt_id = self.queue_prompt(workflow)
        logger.info(f"LatentSync job queued (Prompt ID: {prompt_id}), processing video...")

        outputs = self.wait_for_prompt(prompt_id, timeout=900.0)

        # Output node 6 saves the video
        video_info = outputs.get("6", {}).get("videos", outputs.get("6", {}).get("video", []))[0]
        return self.download_file(
            video_info["filename"],
            subfolder=video_info.get("subfolder", ""),
            folder_type=video_info.get("type", "output"),
            output_path=output_path,
        )

    def voice_convert_cosyvoice3(
        self,
        source_audio_path: str | Path,
        target_audio_path: str | Path,
        output_path: str | Path,
        model_version: str = "Fun-CosyVoice3-0.5B",
        speed: float = 1.0,
        seed: int = 42,
    ) -> Path:  # pragma: no cover
        """Convert source voice timbre to match target voice using CosyVoice 3 on ComfyUI."""
        src_name = self.upload_file(source_audio_path)
        tgt_name = self.upload_file(target_audio_path)
        workflow = self.build_cosyvoice3_vc_workflow(
            source_audio_filename=src_name,
            target_audio_filename=tgt_name,
            model_version=model_version,
            speed=speed,
            seed=seed,
        )
        prompt_id = self.queue_prompt(workflow)
        outputs = self.wait_for_prompt(prompt_id, timeout=120.0)

        audio_info = outputs["5"]["audio"][0]
        return self.download_file(
            audio_info["filename"],
            subfolder=audio_info.get("subfolder", ""),
            folder_type=audio_info.get("type", "output"),
            output_path=output_path,
        )
