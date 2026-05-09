import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

import logging

import main as pipeline
from src.utils import get_device

# Silence SpeechBrain noise
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("speechbrain.utils.importutils").setLevel(logging.ERROR)

st.set_page_config(page_title="Auto-Dub AI", layout="wide")

st.title("🎙️ Auto-Dub: Fully Local AI Video Dubbing")
st.markdown("""
This tool uses a multi-stage AI pipeline to dub videos into other languages:
1. **Vocal Isolation** (Demucs)
2. **Transcription & Diarization** (WhisperX)
3. **Multimodal Translation** (Gemma 4 via Ollama)
4. **Voice Cloning & Synthesis** (Coqui XTTS v2)
5. **Vocal Ducking & Remuxing** (pydub & FFmpeg)
""")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    target_lang = st.selectbox(
        "Target Language",
        [
            "English",
            "Spanish",
            "French",
            "German",
            "Italian",
            "Portuguese",
            "Polish",
            "Turkish",
            "Russian",
            "Dutch",
            "Czech",
            "Arabic",
            "Chinese",
            "Japanese",
            "Korean",
            "Hungarian",
            "Hindi",
        ],
        index=1,
    )

    # Device Selection
    auto_device = get_device()
    device = st.selectbox(
        "Processing Device",
        ["cuda", "mps", "cpu"],
        index=["cuda", "mps", "cpu"].index(auto_device)
        if auto_device in ["cuda", "mps", "cpu"]
        else 2,
    )

    st.divider()
    st.subheader("External Services")
    hf_token = st.text_input(
        "Hugging Face Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Required for speaker diarization (WhisperX)",
    )

    ollama_url = st.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_URL", "http://192.168.86.157:11434"),
        help="The address of your local or remote Ollama instance",
    )
    ollama_model = st.text_input(
        "Ollama Translation Model",
        value=os.getenv("OLLAMA_MODEL", "gemma4:26b"),
        help="Text translation model (e.g. gemma4:26b)",
    )
    ollama_audio_model = st.text_input(
        "Ollama Audio Model",
        value=os.getenv("OLLAMA_AUDIO_MODEL", "gemma4:e4b"),
        help="Audio-informed emotion tagging model. Must be an E-variant of Gemma 4 (e2b/e4b).",
    )

    engine = st.selectbox(
        "Synthesis Engine",
        ["xtts", "fish"],
        index=0,
        help="xtts: Fast zero-shot cloning (24kHz). fish: High-fidelity (44.1kHz) with emotion tagging (requires s2.cpp).",
    )

    col1, col2 = st.columns(2)
    with col1:
        min_speakers = st.number_input(
            "Min Speakers", min_value=1, value=1, step=1, help="Force minimum number of speakers"
        )
    with col2:
        max_speakers = st.number_input(
            "Max Speakers", min_value=1, value=10, step=1, help="Limit maximum number of speakers"
        )

    st.divider()
    st.subheader("Synthesis Quality (Fish Only)")
    tts_temp = st.slider(
        "Temperature", 0.1, 1.0, 0.7, 0.05, help="Higher = more variation, less accent mimicry"
    )
    tts_top_p = st.slider("Top-P", 0.1, 1.0, 0.8, 0.05, help="Higher = more diverse vocabulary")

    # Update environment variables for the pipeline call
    os.environ["OLLAMA_URL"] = ollama_url
    os.environ["OLLAMA_MODEL"] = ollama_model
    os.environ["OLLAMA_AUDIO_MODEL"] = ollama_audio_model

    st.divider()
    st.info(f"Optimal hardware: **{auto_device.upper()}**")

    with st.expander("System Environment"):
        st.code(
            f"""
OS: {os.uname().sysname}
Device: {device}
Ollama: {ollama_url}
Model: {ollama_model}
Audio Model: {ollama_audio_model}
Engine: {engine}
Min Speakers: {min_speakers}
Max Speakers: {max_speakers}
TTS Temp: {tts_temp}
TTS Top-P: {tts_top_p}
        """.strip()
        )

# Main area
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to temp location
    temp_dir = Path("output") / "uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / uploaded_file.name

    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(str(video_path))

    if st.button("🚀 Start Dubbing"):
        log_container = st.empty()

        try:
            with st.status("Dubbing in progress...", expanded=True) as status:
                st.write(f"Step 1: Initializing components on {device}...")
                pipeline.main(
                    str(video_path),
                    target_lang=target_lang,
                    hf_token=hf_token,
                    device=device,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    ollama_audio_model=ollama_audio_model,
                    engine=engine,
                    min_speakers=int(min_speakers),
                    max_speakers=int(max_speakers),
                    tts_temp=float(tts_temp),
                    tts_top_p=float(tts_top_p),
                )

                status.update(label="Dubbing complete!", state="complete", expanded=False)

            st.success("Dubbing finished successfully!")

            # Show the output video
            video_name = Path(video_path).stem
            output_video_path = Path("output") / video_name / f"dubbed_{uploaded_file.name}"

            if output_video_path.exists():
                st.header("Dubbed Result")
                st.video(str(output_video_path))

                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Dubbed Video",
                        data=f,
                        file_name=f"dubbed_{uploaded_file.name}",
                        mime="video/mp4",
                    )
            else:
                st.error("Could not find the output video file.")

        except Exception as e:
            st.error(f"An error occurred during dubbing: {e}")
            st.exception(e)

# Display previously processed videos if any
st.divider()
st.header("Project History")
output_dir = Path("output")
if output_dir.exists():
    projects = [
        p for p in output_dir.iterdir() if p.is_dir() and p.name != "uploads" and p.name != "temp"
    ]
    if projects:
        cols = st.columns(3)
        for i, project in enumerate(projects):
            with cols[i % 3]:
                st.subheader(project.name)
                dubbed_videos = list(project.glob("dubbed_*"))
                if dubbed_videos:
                    st.write(f"Dubbed: {dubbed_videos[0].name}")
                else:
                    st.write("Processing incomplete")
    else:
        st.write("No projects found yet.")
