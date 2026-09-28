from src.comfy_client import ComfyClient, parse_whisperx_srt


def test_comfy_client_init_defaults(monkeypatch):
    monkeypatch.delenv("COMFYUI_URL", raising=False)
    client = ComfyClient()
    assert client.base_url == "http://localhost:8188"
    assert client.timeout == 60.0


def test_comfy_client_init_custom():
    client = ComfyClient(base_url="http://custom-host:8188/", timeout=30.0)
    assert client.base_url == "http://custom-host:8188"
    assert client.timeout == 30.0


def test_comfy_client_init_env(monkeypatch):
    monkeypatch.setenv("COMFYUI_URL", "http://env-host:9999/")
    client = ComfyClient()
    assert client.base_url == "http://env-host:9999"


def test_build_separator_workflow():
    client = ComfyClient()
    wf = client.build_separator_workflow(
        "audio.wav",
        model_filename="test_roformer.ckpt",
        window_size=384,
        aggression=15,
        batch_size=2,
        normalization=0.8,
    )
    assert "1" in wf and wf["1"]["class_type"] == "LoadAudio"
    assert wf["1"]["inputs"]["audio"] == "audio.wav"

    assert "2" in wf and wf["2"]["class_type"] == "AudioSeparatorNode"
    assert wf["2"]["inputs"]["model_filename"] == "test_roformer.ckpt"
    assert wf["2"]["inputs"]["window_size"] == 384
    assert wf["2"]["inputs"]["aggression"] == 15
    assert wf["2"]["inputs"]["batch_size"] == 2
    assert wf["2"]["inputs"]["normalization"] == 0.8

    assert "3" in wf and wf["3"]["class_type"] == "SaveAudio"
    assert "4" in wf and wf["4"]["class_type"] == "SaveAudio"


def test_build_f5_tts_workflow():
    client = ComfyClient()
    wf = client.build_f5_tts_workflow(
        sample_audio_filename="ref.wav",
        sample_text="Hello world",
        speech_text="Translated line",
        speed=1.1,
        seed=100,
        model="F5",
        model_type="F5TTS_Base",
    )
    assert wf["1"]["class_type"] == "LoadAudio"
    assert wf["1"]["inputs"]["audio"] == "ref.wav"

    assert wf["2"]["class_type"] == "F5TTSAudioInputs"
    assert wf["2"]["inputs"]["sample_text"] == "Hello world"
    assert wf["2"]["inputs"]["speech"] == "Translated line"
    assert wf["2"]["inputs"]["speed"] == 1.1
    assert wf["2"]["inputs"]["seed"] == 100
    assert wf["2"]["inputs"]["model"] == "F5"
    assert wf["2"]["inputs"]["model_type"] == "F5TTS_Base"

    assert wf["3"]["class_type"] == "SaveAudio"


def test_build_f5_tts_workflow_fallback_text():
    client = ComfyClient()
    wf = client.build_f5_tts_workflow(
        sample_audio_filename="ref.wav",
        sample_text="",
        speech_text="Test",
    )
    assert wf["2"]["inputs"]["sample_text"] == "Dialogue snippet"


def test_build_latentsync_workflow():
    client = ComfyClient()
    wf = client.build_latentsync_workflow(
        video_filename="video.mp4",
        audio_filename="dub.wav",
        seed=4321,
        inference_steps=25,
        lips_expression=1.8,
    )
    assert wf["1"]["class_type"] == "LoadVideo"
    assert wf["1"]["inputs"]["file"] == "video.mp4"

    assert wf["2"]["class_type"] == "GetVideoComponents"
    assert wf["3"]["class_type"] == "LoadAudio"
    assert wf["3"]["inputs"]["audio"] == "dub.wav"

    assert wf["4"]["class_type"] == "LatentSyncNode"
    assert wf["4"]["inputs"]["seed"] == 4321
    assert wf["4"]["inputs"]["inference_steps"] == 25
    assert wf["4"]["inputs"]["lips_expression"] == 1.8

    assert wf["5"]["class_type"] == "CreateVideo"
    assert wf["6"]["class_type"] == "SaveVideo"


def test_build_demucs_workflow():
    client = ComfyClient()
    wf = client.build_demucs_workflow(
        "song.wav",
        model="💾 Hybrid Transformer",
        shifts=2,
        overlap=0.3,
        target_device="cuda",
    )
    assert wf["1"]["class_type"] == "LoadAudio"
    assert wf["1"]["inputs"]["audio"] == "song.wav"

    assert wf["2"]["class_type"] == "AudioSeparateDemucs"
    assert wf["2"]["inputs"]["model"] == "💾 Hybrid Transformer"
    assert wf["2"]["inputs"]["shifts"] == 2
    assert wf["2"]["inputs"]["overlap"] == 0.3
    assert wf["2"]["inputs"]["target_device"] == "cuda"

    assert wf["3"]["class_type"] == "SaveAudio"
    assert wf["4"]["class_type"] == "AudioMerge"
    assert wf["5"]["class_type"] == "AudioMerge"
    assert wf["6"]["class_type"] == "SaveAudio"


def test_build_whisperx_workflow():
    client = ComfyClient()
    wf = client.build_whisperx_workflow(
        "audio.wav",
        model_type="large-v3",
        batch_size=8,
        if_mutiple_speaker=True,
    )
    assert wf["1"]["class_type"] == "LoadAudioVideoPath"
    assert wf["1"]["inputs"]["audio"] == "audio.wav"

    assert wf["2"]["class_type"] == "WhisperX"
    assert wf["2"]["inputs"]["model_type"] == "large-v3"
    assert wf["2"]["inputs"]["batch_size"] == 8
    assert wf["2"]["inputs"]["if_mutiple_speaker"] is True

    assert wf["3"]["class_type"] == "PreViewSRT"


def test_parse_whisperx_srt():
    srt = """1
00:00:01,000 --> 00:00:02,500
0 Hello world.

2
00:00:03,000 --> 00:00:04,200
1 How are you?

3
00:00:05,000 --> 00:00:06,000
No speaker number here.
"""
    segs = parse_whisperx_srt(srt)
    assert len(segs) == 3
    assert segs[0] == {"start": 1.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "Hello world."}
    assert segs[1] == {"start": 3.0, "end": 4.2, "speaker": "SPEAKER_01", "text": "How are you?"}
    assert segs[2] == {
        "start": 5.0,
        "end": 6.0,
        "speaker": "SPEAKER_00",
        "text": "No speaker number here.",
    }


def test_build_cosyvoice3_vc_workflow():
    client = ComfyClient()
    wf = client.build_cosyvoice3_vc_workflow(
        source_audio_filename="src.wav",
        target_audio_filename="tgt.wav",
        model_version="Fun-CosyVoice3-0.5B",
        speed=1.0,
        seed=123,
    )
    assert wf["1"]["class_type"] == "FL_CosyVoice3_ModelLoader"
    assert wf["1"]["inputs"]["model_version"] == "Fun-CosyVoice3-0.5B"
    assert wf["2"]["class_type"] == "LoadAudio"
    assert wf["2"]["inputs"]["audio"] == "src.wav"
    assert wf["3"]["class_type"] == "LoadAudio"
    assert wf["3"]["inputs"]["audio"] == "tgt.wav"
    assert wf["4"]["class_type"] == "FL_CosyVoice3_VoiceConversion"
    assert wf["4"]["inputs"]["speed"] == 1.0
    assert wf["4"]["inputs"]["seed"] == 123
    assert wf["5"]["class_type"] == "SaveAudio"
