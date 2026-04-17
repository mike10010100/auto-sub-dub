import os
import json
import base64
from pathlib import Path
from tqdm import tqdm
from ollama import Client

class Translator:
    def __init__(self, ollama_url="http://192.168.86.172:11434", model="gemma4:e4b"):
        self.client = Client(host=ollama_url)
        self.model = model

    def translate_segments_multimodal(self, segments, vocals_path, target_lang="English"):
        """
        Translates each segment using the multimodal capabilities of Gemma 4.
        Feeds both the original transcript text AND the raw audio of the segment
        to the LLM to capture emotional context and prosody.
        """
        from pydub import AudioSegment
        
        print(f"Translating {len(segments)} segments to {target_lang} (Hybrid Multimodal)...")
        audio = AudioSegment.from_wav(vocals_path)
        
        translated_segments = []
        
        system_prompt = (
            f"You are a professional video translator and voice director. Your task is to translate subtitles "
            f"from their source language into {target_lang} while strictly matching the performance of the audio provided.\n\n"
            "INSTRUCTIONS:\n"
            f"1. TRANSLATION: Translate the text to {target_lang}. Ensure it is natural and matches the speaker's intent.\n"
            "2. TIMING: The translation MUST be speakable within the original timeframe. Use concise language for short clips.\n"
            "3. EMOTION: Analyze the audio clip for emotional cues (e.g., sarcasm, anger, whispering, excitement).\n"
            "4. OUTPUT FORMAT: Respond with ONLY a JSON object in this format: "
            "{\"translated_text\": \"...\", \"emotion\": \"[EMOTION_TAG]\"}\n"
            "Valid tags: [NEUTRAL], [WHISPER], [ANGRY], [EXCITED], [SAD], [SARCASM], [FRIENDLY], [SHOUTING]."
        )
        
        for i, segment in enumerate(tqdm(segments, desc="Translating with Audio")):
            original_text = segment.get("text", "")
            duration = segment.get("end", 0) - segment.get("start", 0)
            
            if not original_text.strip():
                translated_segments.append(segment)
                continue
            
            # Extract the specific audio clip for this segment
            start_ms = int(segment["start"] * 1000)
            end_ms = int(segment["end"] * 1000)
            clip = audio[start_ms:end_ms]
            
            # Export to a temporary buffer
            import io
            buffer = io.BytesIO()
            clip.export(buffer, format="wav")
            audio_bytes = buffer.getvalue()
            
            try:
                # Call Ollama Multimodal API
                response = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user", 
                            "content": f"Translate this segment (Target Duration: {duration:.2f}s). Original Transcript: '{original_text}'",
                            "audio": [audio_bytes]
                        }
                    ],
                    options={"temperature": 0.2}
                )
                
                content = response['message']['content'].strip()
                
                # Attempt to parse JSON from the response
                try:
                    # Look for JSON block if model was wordy
                    if "{" in content and "}" in content:
                        content = content[content.find("{"):content.rfind("}")+1]
                    
                    data = json.loads(content)
                    translated_text = data.get("translated_text", "").strip()
                    emotion = data.get("emotion", "[NEUTRAL]")
                except (json.JSONDecodeError, ValueError):
                    # Fallback if JSON parsing fails
                    translated_text = content
                    emotion = "[NEUTRAL]"
                
                new_segment = segment.copy()
                new_segment["original_text"] = original_text
                new_segment["text"] = translated_text
                new_segment["emotion"] = emotion
                translated_segments.append(new_segment)
                
            except Exception as e:
                print(f"Ollama multimodal translation failed for segment {i}: {e}")
                translated_segments.append(segment)
        
        return translated_segments

if __name__ == "__main__":
    # Simple test stub
    print("Translator Multimodal module loaded.")
