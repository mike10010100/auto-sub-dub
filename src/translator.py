import os
import json
import base64
from pathlib import Path
from tqdm import tqdm
from ollama import Client
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class Translator:
    def __init__(self, ollama_url=None, model=None):
        # Priority: argument > environment variable > default
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://192.168.86.172:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma4")
        
        self.client = Client(host=self.ollama_url)
        logger.info(f"Initialized Translator with Ollama at {self.ollama_url} (model: {self.model})")

    def translate_segments_multimodal(self, segments, vocals_path, target_lang="English", max_workers=4):
        """
        Translates each segment using the multimodal capabilities of Gemma 4.
        Feeds both the original transcript text AND the raw audio of the segment
        to the LLM to capture emotional context and prosody.
        """
        from pydub import AudioSegment
        
        logger.info(f"Translating {len(segments)} segments to {target_lang} (Gemma 4 Multimodal, max_workers={max_workers})...")
        audio = AudioSegment.from_wav(vocals_path)
        
        system_prompt = (
            "<|think|>You are a professional video translator and voice director. Your task is to translate subtitles "
            f"from their source language into {target_lang} while strictly matching the performance of the audio provided.\n\n"
            "INSTRUCTIONS:\n"
            f"1. TRANSLATION: Translate the text to {target_lang}. Ensure it is natural and matches the speaker's intent.\n"
            "2. TIMING: The translation MUST be speakable within the original timeframe. Use concise language for short clips.\n"
            "3. EMOTION: Analyze the audio clip for emotional cues (e.g., sarcasm, anger, whispering, excitement).\n"
            "4. OUTPUT FORMAT: Respond with ONLY a JSON object (after your thinking block) in this format: "
            "{\"translated_text\": \"...\", \"emotion\": \"[EMOTION_TAG]\"}\n"
            "Valid tags: [NEUTRAL], [WHISPER], [ANGRY], [EXCITED], [SAD], [SARCASM], [FRIENDLY], [SHOUTING]."
        )

        def process_segment(index_segment):
            idx, segment = index_segment
            original_text = segment.get("text", "")
            duration = segment.get("end", 0) - segment.get("start", 0)
            
            if not original_text.strip():
                return idx, segment
            
            # Extract the specific audio clip for this segment
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            clip = audio[start_ms:end_ms]
            
            import io
            buffer = io.BytesIO()
            clip.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": f"[AUDIO INPUT] Translate this segment (Target Duration: {duration:.2f}s). Original Transcript: '{original_text}'",
                            "audio": [audio_bytes]
                        }
                    ],
                    options={
                        "temperature": 1.0,
                        "top_p": 0.95,
                        "top_k": 64
                    }
                )
                
                content = response['message']['content'].strip()
                
                if "<channel|>" in content:
                    content = content.split("<channel|>")[-1].strip()
                
                try:
                    if "{" in content and "}" in content:
                        content = content[content.find("{"):content.rfind("}")+1]
                    
                    data = json.loads(content)
                    translated_text = data.get("translated_text", "").strip()
                    emotion = data.get("emotion", "[NEUTRAL]")
                except (json.JSONDecodeError, ValueError):
                    translated_text = content
                    emotion = "[NEUTRAL]"
                
                new_segment = segment.copy()
                new_segment["original_text"] = original_text
                new_segment["text"] = translated_text
                new_segment["emotion"] = emotion
                return idx, new_segment
                
            except Exception as e:
                logger.error(f"Ollama multimodal translation failed for segment {idx}: {e}")
                return idx, segment

        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_segments = list(enumerate(segments))
            results = list(tqdm(executor.map(process_segment, indexed_segments), total=len(segments), desc="Translating with Audio"))
        
        # Sort results by original index and extract segments
        results.sort(key=lambda x: x[0])
        translated_segments = [res[1] for res in results]
        
        return translated_segments

if __name__ == "__main__":
    # Simple test stub
    logging.basicConfig(level=logging.INFO)
    logger.info("Translator Multimodal module loaded.")
