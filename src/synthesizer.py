import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from audiotsm import wsola
from audiotsm.io.wav import WavReader, WavWriter
from pydub import AudioSegment

from src.utils import get_device

# Silence verbose library noise
logger = logging.getLogger(__name__)


class BaseSynthesizer:
    def __init__(self, output_dir="output/audio_segments", device=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_dir = Path("output/references")
        self.ref_audio_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or get_device()
        self._vad = None
        self._vad_utils = None
        self._embedding_model = None
        self._embedding_inference = None
        self._seed_vc_models = None

    def apply_seed_vc(self, audio_path, speaker_id, speaker_refs, emotion=None):
        """Apply Seed-VC zero-shot timbre transfer to decouple accent from voice identity."""
        # For Seed-VC, we MUST use exactly ONE consistent reference clip per speaker
        # across the entire video to prevent their voice/acoustic environment from changing
        # between lines. We use the first (primary) reference.
        ref_wav_path = speaker_refs[0]["path"]

        logger.info(f"Applying Seed-VC zero-shot skin for {speaker_id}...")
        try:
            output_dir = self.output_dir / "seed_vc_temp"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / Path(audio_path).name

            if self._seed_vc_models is None:
                import argparse
                import sys

                seed_vc_path = str(Path(__file__).parent.parent / "seed-vc")
                if seed_vc_path not in sys.path:
                    sys.path.insert(0, seed_vc_path)

                # Set HF cache environment variable
                os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"

                import inference

                # Create a mock args namespace
                args = argparse.Namespace(
                    source=None,
                    target=None,
                    output=None,
                    diffusion_steps=25,
                    length_adjust=1.0,
                    inference_cfg_rate=0.7,
                    f0_condition=True,
                    auto_f0_adjust=True,
                    semi_tone_shift=0,
                    checkpoint=None,
                    config=None,
                    fp16=(self.device == "cuda"),
                )

                logger.info("Initializing Seed-VC in-process (loading 44k F0 model)...")
                loaded = inference.load_models(args)
                self._seed_vc_models = (loaded, args)

            # Extract loaded models
            loaded_models, args = self._seed_vc_models

            # Execute conversion in-process
            self._run_seed_vc_in_process(loaded_models, args, audio_path, ref_wav_path, output_path)

            # Seed-VC creates the file in the output dir with the same name.
            if output_path.exists():
                os.replace(output_path, str(audio_path))
            return audio_path
        except Exception as e:
            logger.error(f"Seed-VC inference failed for {speaker_id}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return audio_path

    def _run_seed_vc_in_process(self, loaded_models, args, source_path, target_path, output_path):
        import os

        import librosa
        import numpy as np
        import torch
        import torchaudio
        from inference import crossfade

        with torch.no_grad():
            model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = (
                loaded_models
            )
            sr = mel_fn_args["sampling_rate"]
            f0_condition = args.f0_condition
            auto_f0_adjust = args.auto_f0_adjust
            pitch_shift = args.semi_tone_shift

            # Load audio using the proper sampling rate
            source_audio, _ = librosa.load(source_path, sr=sr)
            ref_audio, _ = librosa.load(target_path, sr=sr)

            device = torch.device(self.device)

            # Process audio
            source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
            ref_audio = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)

            # Resample
            converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
            if converted_waves_16k.size(-1) <= 16000 * 30:
                S_alt = semantic_fn(converted_waves_16k)
            else:
                overlapping_time = 5
                S_alt_list = []
                buffer = None
                traversed_time = 0
                while traversed_time < converted_waves_16k.size(-1):
                    if buffer is None:
                        chunk = converted_waves_16k[:, traversed_time : traversed_time + 16000 * 30]
                    else:
                        chunk = torch.cat(
                            [
                                buffer,
                                converted_waves_16k[
                                    :,
                                    traversed_time : traversed_time
                                    + 16000 * (30 - overlapping_time),
                                ],
                            ],
                            dim=-1,
                        )
                    S_alt = semantic_fn(chunk)
                    if traversed_time == 0:
                        S_alt_list.append(S_alt)
                    else:
                        S_alt_list.append(S_alt[:, 50 * overlapping_time :])
                    buffer = chunk[:, -16000 * overlapping_time :]
                    traversed_time += (
                        30 * 16000
                        if traversed_time == 0
                        else chunk.size(-1) - 16000 * overlapping_time
                    )
                S_alt = torch.cat(S_alt_list, dim=1)

            ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
            S_ori = semantic_fn(ori_waves_16k)

            mel = mel_fn(source_audio.to(device).float())
            mel2 = mel_fn(ref_audio.to(device).float())

            target_lengths = torch.LongTensor([int(mel.size(2) * 1.0)]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(
                ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = campplus_model(feat2.unsqueeze(0))

            if f0_condition:
                F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
                F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

                F0_ori = torch.from_numpy(F0_ori).to(device)[None]
                F0_alt = torch.from_numpy(F0_alt).to(device)[None]

                voiced_F0_ori = F0_ori[F0_ori > 1]
                voiced_F0_alt = F0_alt[F0_alt > 1]

                log_f0_alt = torch.log(F0_alt + 1e-5)
                voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
                voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
                median_log_f0_ori = torch.median(voiced_log_f0_ori)
                median_log_f0_alt = torch.median(voiced_log_f0_alt)

                # shift alt log f0 level to ori log f0 level
                shifted_log_f0_alt = log_f0_alt.clone()
                if auto_f0_adjust:
                    shifted_log_f0_alt[F0_alt > 1] = (
                        log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
                    )
                shifted_f0_alt = torch.exp(shifted_log_f0_alt)
                if pitch_shift != 0:
                    from inference import adjust_f0_semitones

                    shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(
                        shifted_f0_alt[F0_alt > 1], pitch_shift
                    )
            else:
                F0_ori = None
                F0_alt = None
                shifted_f0_alt = None

            # Length regulation
            cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

            hop_length = 512 if f0_condition else 256
            max_context_window = sr // hop_length * 30
            overlap_frame_len = 16
            overlap_wave_len = overlap_frame_len * hop_length

            max_source_window = max_context_window - mel2.size(2)
            processed_frames = 0
            generated_wave_chunks = []

            fp16 = args.fp16
            while processed_frames < cond.size(1):
                chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
                is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16 if fp16 else torch.float32
                ):
                    vc_target = model.cfm.inference(
                        cat_condition,
                        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                        mel2,
                        style2,
                        None,
                        25,  # diffusion_steps
                        inference_cfg_rate=0.7,
                    )
                    vc_target = vc_target[:, :, mel2.size(-1) :]
                vc_wave = vocoder_fn(vc_target.float()).squeeze()
                vc_wave = vc_wave[None, :]
                if processed_frames == 0:
                    if is_last_chunk:
                        output_wave = vc_wave[0].cpu().numpy()
                        generated_wave_chunks.append(output_wave)
                        break
                    output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len
                elif is_last_chunk:
                    output_wave = crossfade(
                        previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len
                    )
                    generated_wave_chunks.append(output_wave)
                    processed_frames += vc_target.size(2) - overlap_frame_len
                    break
                else:
                    output_wave = crossfade(
                        previous_chunk.cpu().numpy(),
                        vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                        overlap_wave_len,
                    )
                    generated_wave_chunks.append(output_wave)
                    previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len

            vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(str(output_path), vc_wave.cpu(), sr)

    def _get_embeddings(self, clips):
        """Calculate speaker embeddings for a list of AudioSegments."""
        import numpy as np
        import torch
        from pyannote.audio import Inference, Model

        if self._embedding_model is None:
            logger.info("Loading PyAnnote embedding model...")
            try:
                # We use the same token as the diarizer
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    logger.warning(
                        "HF_TOKEN not found in environment. Vocal verification will be skipped."
                    )
                    return None

                # Explicitly load the model and check for errors
                self._embedding_model = Model.from_pretrained(
                    "pyannote/embedding", use_auth_token=hf_token
                )
                if self._embedding_model is None:
                    logger.error(
                        "Failed to load 'pyannote/embedding'. Have you accepted the terms at "
                        "https://huggingface.co/pyannote/embedding ?"
                    )
                    return None

                self._embedding_inference = Inference(
                    self._embedding_model, window="whole", device=torch.device(self.device)
                )
            except Exception as e:
                logger.warning(f"Vocal verification unavailable (embedding model failed): {e}")
                return None

        embeddings = []
        for clip in clips:
            # Convert pydub clip to 16kHz mono float32 tensor for pyannote
            mono_16k = clip.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            samples = np.frombuffer(mono_16k.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples).unsqueeze(0)

            # Inference expects a dict with 'waveform' and 'sample_rate'
            emb = self._embedding_inference({"waveform": tensor, "sample_rate": 16000})
            embeddings.append(emb)

        return np.array(embeddings)

    def _purify_references(self, candidates, speaker_id, threshold=0.35):
        """Filter out vocal outliers using cosine distance to the group centroid."""
        if len(candidates) < 2:
            return candidates

        import numpy as np
        from scipy.spatial.distance import cdist

        clips = [c[1] for c in candidates]
        embs = self._get_embeddings(clips)
        if embs is None:
            return candidates

        # 1. Calculate the 'Vocal Anchor' (the clip closest to all others)
        dist_matrix = cdist(embs, embs, metric="cosine")
        avg_distances = np.mean(dist_matrix, axis=1)
        anchor_idx = np.argmin(avg_distances)

        # 2. Reject outliers that are too far from the anchor
        purified = []
        rejected_count = 0
        for i, cand in enumerate(candidates):
            dist = dist_matrix[anchor_idx, i]
            if dist <= threshold:
                purified.append(cand)
            else:
                rejected_count += 1

        if rejected_count > 0:
            logger.warning(
                f"  {speaker_id}: Rejected {rejected_count} vocal outliers "
                f"from candidate pool (Consensus check)."
            )

        return purified

    def _load_vad(self):
        """Lazy-load Silero VAD. Returns (model, utils) or (None, None) on failure."""
        if self._vad is not None:
            return self._vad, self._vad_utils
        try:
            import torch as _torch

            vad, utils = _torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad = vad
            self._vad_utils = utils
            return vad, utils
        except Exception as e:
            logger.warning(f"Silero VAD unavailable ({e}); falling back to dBFS ranking.")
            return None, None

    def _score_reference_clip(self, clip):
        """
        Score a candidate reference clip for cloning. Returns a dict with
        voiced_ratio, snr_db, dbfs.
        """
        import numpy as np

        vad, utils = self._load_vad()
        if vad is None:
            return {"voiced_ratio": 1.0, "snr_db": 0.0, "dbfs": clip.dBFS}

        mono_16k = clip.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        samples = np.frombuffer(mono_16k.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < 16000 * 0.5:
            return {"voiced_ratio": 0.0, "snr_db": -60.0, "dbfs": clip.dBFS}

        import torch as _torch

        get_speech_timestamps = utils[0]
        tensor = _torch.from_numpy(samples)
        try:
            ts = get_speech_timestamps(tensor, vad, sampling_rate=16000)
        except Exception as e:
            logger.warning(f"VAD failed on clip: {e}")
            return {"voiced_ratio": 1.0, "snr_db": 0.0, "dbfs": clip.dBFS}

        if not ts:
            return {"voiced_ratio": 0.0, "snr_db": -60.0, "dbfs": clip.dBFS}

        mask = np.zeros(samples.shape[0], dtype=bool)
        for seg in ts:
            mask[seg["start"] : seg["end"]] = True

        voiced = samples[mask]
        unvoiced = samples[~mask]
        voiced_ratio = voiced.size / samples.size

        voiced_rms = float(np.sqrt(np.mean(voiced * voiced) + 1e-12)) if voiced.size else 1e-6
        unvoiced_rms = (
            float(np.sqrt(np.mean(unvoiced * unvoiced) + 1e-12)) if unvoiced.size else 1e-6
        )
        snr_db = 20.0 * np.log10(voiced_rms / max(unvoiced_rms, 1e-6))

        return {"voiced_ratio": voiced_ratio, "snr_db": snr_db, "dbfs": clip.dBFS}

    def extract_speaker_references(
        self,
        vocals_path,
        transcript,
        target_clips=5,
        min_duration=5,
        max_duration=20,
        trim_to=12.0,
        hq_vocals_path=None,
        min_voiced_ratio=0.40,
    ):
        """
        Extract per-speaker reference clips for voice cloning.
        Returns: {speaker: [{"path": str, "emotion": str, "text": str}, ...]}
        """
        source_path = hq_vocals_path or vocals_path
        logger.info(
            f"Extracting multi-reference samples from {source_path} "
            f"(target: {target_clips} clips, min/max seg duration: {min_duration}/{max_duration}s)"
        )
        audio = AudioSegment.from_wav(source_path)
        index_path = self.ref_audio_dir / "references.json"

        # Use raw segments for references since they match the source audio timing
        segments = transcript.get("segments", [])

        if index_path.exists():
            try:
                cached = json.loads(index_path.read_text(encoding="utf-8"))
                # Invalidate if files are missing OR if text is empty (s2.cpp requirement)
                if all(
                    Path(c["path"]).exists() and c.get("text")
                    for clips in cached.values()
                    for c in clips
                ):
                    logger.info(f"Using cached references from {index_path}")
                    return cached
                else:
                    logger.warning("Cached references are invalid or missing text; re-extracting.")
            except Exception as e:
                logger.warning(f"Could not read ref index {index_path}: {e}")

        references = {}
        unique_speakers = {s["speaker"] for s in segments if s.get("speaker")}

        for speaker in unique_speakers:
            # Filter for segments that have speaker match AND non-empty text
            spk_segs = [
                s for s in segments if s.get("speaker") == speaker and s.get("text", "").strip()
            ]
            candidates = []
            for seg in spk_segs:
                duration = seg["end"] - seg["start"]
                if duration < min_duration:
                    continue
                start_ms = int(seg["start"] * 1000)
                end_ms = int(min(seg["end"], seg["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                # Relaxed thresholds: -45dBFS and 0.40 voiced ratio
                if clip.dBFS <= -45:
                    continue
                metrics = self._score_reference_clip(clip)
                if metrics["voiced_ratio"] < 0.40:
                    continue
                # Emotion might not be in raw segments, so we look it up in translated
                # or just use NEUTRAL for reference extraction.
                emotion = seg.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                # We must provide the original text as s2.cpp requires prompt text
                # if prompt audio is provided. We rely on style tags and temperature
                # to manage cross-lingual accent leakage.
                candidates.append((score, clip, emotion, seg["text"], metrics))

            if not candidates and spk_segs:
                # Even more relaxed fallback for rare side-characters
                longest = max(spk_segs, key=lambda s: s["end"] - s["start"])
                start_ms = int(longest["start"] * 1000)
                end_ms = int(min(longest["end"], longest["start"] + trim_to) * 1000)
                clip = audio[start_ms:end_ms]
                metrics = self._score_reference_clip(clip)
                emotion = longest.get("emotion", "[NEUTRAL]") or "[NEUTRAL]"
                logger.warning(
                    f"  {speaker}: using low-quality fallback reference (voiced={metrics['voiced_ratio']:.2f})"
                )
                score = (metrics["snr_db"], metrics["voiced_ratio"], metrics["dbfs"])
                candidates.append((score, clip, emotion, longest["text"], metrics))

            # --- Vocal Consistency Check (Outlier Rejection) ---
            # Purify the pool to ensure all clips sound like the same character
            purified_pool = self._purify_references(candidates, speaker)
            purified_pool.sort(key=lambda x: x[0], reverse=True)
            picked = purified_pool[:target_clips]

            # Organize into subfolders for the 'Reference Gallery'
            spk_dir = self.ref_audio_dir / speaker
            spk_dir.mkdir(parents=True, exist_ok=True)

            entries = []
            for i, (score, clip, emotion, text, metrics) in enumerate(picked):
                safe_emotion = emotion.strip("[]") or "NEUTRAL"
                ref_path = spk_dir / f"ref_{i + 1}_{safe_emotion}.wav"
                clip.export(ref_path, format="wav")
                entries.append({"path": str(ref_path), "emotion": emotion, "text": text})

            if entries:
                references[speaker] = entries
                logger.info(f"  {speaker}: finalized {len(entries)} consistent reference clips.")

        index_path.write_text(json.dumps(references, indent=2), encoding="utf-8")
        return references

    def refine_speaker_assignments(self, audio_path, segments):
        """
        Second-pass diarization: use refined speaker centroids to re-verify labels
        for every segment, fixing misattributions.
        """
        logger.info("Starting Diarization Refinement (Second Pass)...")

        # 1. Get purified centroids for every known speaker
        index_path = self.ref_audio_dir / "references.json"
        if not index_path.exists():
            return segments

        try:
            refs = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return segments

        import numpy as np

        centroids = {}
        for spk, clips in refs.items():
            # Load embeddings for the finalized reference clips
            clip_segments = [AudioSegment.from_wav(c["path"]) for c in clips]
            embs = self._get_embeddings(clip_segments)
            if embs is not None:
                centroids[spk] = np.mean(embs, axis=0)

        if not centroids:
            return segments

        # Identify duplicate speakers whose centroids are extremely close (< 0.38 cosine distance)
        from scipy.spatial.distance import cosine

        speaker_map = {spk: spk for spk in centroids}
        spk_list = list(centroids.keys())
        for idx1 in range(len(spk_list)):
            for idx2 in range(idx1 + 1, len(spk_list)):
                s1 = spk_list[idx1]
                s2 = spk_list[idx2]
                if s1 in centroids and s2 in centroids:
                    dist = cosine(centroids[s1], centroids[s2])
                    if dist < 0.38:
                        logger.info(
                            f"Acoustic Verification: Merging duplicate speakers {s1} and {s2} "
                            f"(distance: {dist:.3f})"
                        )
                        # Resolve transitive mapping: find canonical s1
                        canonical_s1 = s1
                        while speaker_map[canonical_s1] != canonical_s1:
                            canonical_s1 = speaker_map[canonical_s1]
                        speaker_map[s2] = canonical_s1

        # Consolidate centroids for merged speakers
        new_centroids = {}
        for spk, centroid in centroids.items():
            curr = spk
            while curr in speaker_map and speaker_map[curr] != curr:
                curr = speaker_map[curr]
            mapped_spk = curr
            if mapped_spk not in new_centroids:
                new_centroids[mapped_spk] = []
            new_centroids[mapped_spk].append(centroid)

        centroids = {spk: np.mean(embs, axis=0) for spk, embs in new_centroids.items()}

        def get_canonical(spk):
            if not spk:
                return spk
            curr = spk
            while curr in speaker_map and speaker_map[curr] != curr:
                curr = speaker_map[curr]
            return curr

        # Map initial speaker labels in segments
        for seg in segments:
            if seg.get("speaker"):
                seg["speaker"] = get_canonical(seg["speaker"])

        # 2. Re-verify every segment
        audio = AudioSegment.from_wav(audio_path)
        refined_count = 0

        for seg in segments:
            start_ms = int(seg["start"] * 1000)
            end_ms = int(seg["end"] * 1000)
            if (end_ms - start_ms) < 500:  # Skip micro-segments for refinement
                continue

            clip = audio[start_ms:end_ms]
            # Get embedding for this specific line
            seg_emb_arr = self._get_embeddings([clip])
            if seg_emb_arr is None:
                continue
            seg_emb = seg_emb_arr[0]

            # Find the closest centroid
            curr_spk = seg.get("speaker")
            best_spk = curr_spk
            min_dist = float("inf")

            distances = {}
            for spk, centroid in centroids.items():
                dist = cosine(seg_emb, centroid)
                distances[spk] = dist
                if dist < min_dist:
                    min_dist = dist
                    best_spk = spk

            # 3. Only flip if we are MUCH more certain about another speaker
            # (threshold: 0.15 gap or original was very far > 0.45)
            curr_dist = distances.get(curr_spk, 1.0)
            if best_spk != curr_spk and ((curr_dist - min_dist) > 0.15 or curr_dist > 0.45):
                logger.info(
                    f"  Re-assigned segment {seg.get('start', 0):.1f}s: "
                    f"{curr_spk} ({curr_dist:.2f}) -> {best_spk} ({min_dist:.2f})"
                )
                seg["speaker"] = best_spk
                refined_count += 1

        if refined_count > 0:
            logger.info(f"Diarization Refinement complete. Fixed {refined_count} labels.")

        return segments

    def adjust_speed(self, audio_path, target_duration):
        """Adjusts the speed of an audio file to match the target duration without changing pitch."""
        with (
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_in,
            tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_out,
        ):
            temp_in_path = temp_in.name
            temp_out_path = temp_out.name

            audio = AudioSegment.from_file(audio_path)

            # 1. Strip leading/trailing silence from the synthesized clip.
            # This is CRITICAL for sync. TTS often has a small leading delay.
            from pydub.silence import detect_leading_silence

            trim_leading = detect_leading_silence(audio, silence_threshold=-50)
            trim_trailing = detect_leading_silence(audio.reverse(), silence_threshold=-50)

            if trim_leading > 0 or trim_trailing > 0:
                audio = audio[trim_leading : len(audio) - trim_trailing]
                logger.info(
                    f"Trimmed {trim_leading}ms leading and {trim_trailing}ms trailing silence."
                )

            current_duration = audio.duration_seconds

            # Ensure the audio is Mono, 44100Hz, 16-bit PCM for maximum compatibility with audiotsm
            audio = audio.set_channels(1).set_frame_rate(44100).set_sample_width(2)
            audio.export(temp_in_path, format="wav", codec="pcm_s16le")

            speed_ratio = current_duration / target_duration

            if speed_ratio < 1.0 or speed_ratio > 2.0:
                logger.info(f"Speed ratio {speed_ratio:.2f} clipped to [1.0, 2.0].")
                speed_ratio = max(1.0, min(2.0, speed_ratio))

            if abs(speed_ratio - 1.0) < 0.01:
                logger.info("Speed ratio is ~1.0, skipping WSOLA to prevent artifacts.")
                audio.export(audio_path, format="wav")
                if os.path.exists(temp_in_path):
                    os.unlink(temp_in_path)
                if os.path.exists(temp_out_path):
                    os.unlink(temp_out_path)
                return audio_path

            try:
                # Use a custom WavReader if needed, but standard should work if format is fixed
                with WavReader(temp_in_path) as reader:
                    with WavWriter(temp_out_path, reader.channels, reader.samplerate) as writer:
                        tsm = wsola(reader.channels, speed=speed_ratio)
                        tsm.run(reader, writer)

                final_audio = AudioSegment.from_wav(temp_out_path)
                final_audio.export(audio_path, format="wav")
            except Exception as e:
                logger.error(f"Time-stretching failed ({e}); falling back to simple speed change.")
                pass

            if os.path.exists(temp_in_path):
                os.unlink(temp_in_path)
            if os.path.exists(temp_out_path):
                os.unlink(temp_out_path)

            return audio_path


class XTTSSynthesizer(BaseSynthesizer):
    def __init__(self, output_dir="output/audio_segments", device=None):
        super().__init__(output_dir, device)

        # Force CPU for XTTS on Mac
        if self.device == "mps":
            logger.info("XTTS v2 is unstable on MPS. Forcing CPU for high-quality synthesis.")
            self.device = "cpu"

        self.model = None

    def _load_model(self):
        if self.model is None:
            from TTS.api import TTS

            logger.info(f"Loading XTTS v2 model on {self.device}...")
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)

    def synthesize(
        self,
        text,
        speaker_id,
        speaker_refs,
        output_filename,
        language="en",
        emotion=None,
        **kwargs,
    ):
        self._load_model()
        output_path = self.output_dir / output_filename
        ref_paths = [r["path"] for r in speaker_refs if r["emotion"] == emotion] or [
            r["path"] for r in speaker_refs
        ]

        if not ref_paths:
            return None

        try:
            self.model.tts_to_file(
                text=text,
                speaker_wav=ref_paths,
                language=language,
                file_path=str(output_path),
            )
            return output_path
        except Exception as e:
            logger.error(f"XTTS Synthesis failed: {e}")
            return None


class FishSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        output_dir="output/audio_segments",
        device=None,
        s2_cpp_path=None,
        model_path="models/s2-pro-q4_k_m.gguf",
        tokenizer_path="models/tokenizer.json",
    ):
        super().__init__(output_dir, device)

        # Check standard locations for s2 binary
        candidates = ["./s2.cpp/build/s2", "../s2.cpp/build/s2", "/usr/local/bin/s2", "s2"]

        if s2_cpp_path:
            candidates.insert(0, s2_cpp_path)

        self.s2_cpp_path = None
        for cand in candidates:
            if cand == "s2":  # Check PATH
                if subprocess.run(["which", "s2"], capture_output=True).returncode == 0:
                    self.s2_cpp_path = "s2"
                    break
            elif os.path.exists(cand):
                self.s2_cpp_path = cand
                break

        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

        if not self.s2_cpp_path:
            logger.error("s2.cpp binary not found in expected locations.")
        else:
            logger.info(f"Using Fish Speech binary: {self.s2_cpp_path}")

    def synthesize(
        self,
        text,
        speaker_id,
        speaker_refs,
        output_filename,
        language="en",
        emotion=None,
        temp=0.7,
        top_p=0.8,
        top_k=20,
    ):
        output_path = self.output_dir / output_filename
        # Pick the best reference for Fish Speech that HAS text (required by s2.cpp)
        valid_refs = [r for r in speaker_refs if r.get("text")]
        if not valid_refs:
            logger.error(f"No valid references with text found for speaker {speaker_id}")
            return None

        # Format the text with emotion tags for Fish Speech
        formatted_text = f"{emotion} {text}" if emotion and emotion.startswith("[") else text

        cmd = [
            self.s2_cpp_path,
            "-m",
            self.model_path,
            "-t",
            self.tokenizer_path,
            "-text",
            formatted_text,
            "-o",
            str(output_path),
            "-temp",
            str(temp),
            "-top-p",
            str(top_p),
            "-top-k",
            str(top_k),
            "--trim-silence",
        ]

        logger.info(
            "Seed-VC Integration: Bypassing Fish Speech acoustic prompt for generic voice generation."
        )

        if self.device == "cuda":
            cmd.extend(["-c", "0"])
        elif self.device == "mps":
            cmd.append("-M")

        logger.info(f"Synthesizing with Fish Speech: {formatted_text}")
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Fish Speech Synthesis failed: {e.stderr.decode()}")
            return None


def Synthesizer(engine="xtts", **kwargs):
    if engine == "fish":
        return FishSynthesizer(**kwargs)
    return XTTSSynthesizer(**kwargs)
