import requests
import json

from tqdm import tqdm

class Translator:
    def __init__(self, ollama_url="http://192.168.86.172:11434", model="gemma4"):
        self.ollama_url = f"{ollama_url}/api/chat"
        self.model = model

    def translate_segments(self, segments, target_lang="Spanish"):
        """Translates each segment using the local Ollama instance."""
        print(f"Translating {len(segments)} segments to {target_lang}...")
        
        translated_segments = []
        
        # We'll send segments in chunks to maintain some context while avoiding overwhelming the context window
        # For simplicity, let's start with individual segment translations but with a system prompt that explains the context
        
        system_prompt = (
            f"You are a professional video translator. Your task is to translate subtitles from their source language into {target_lang}. "
            "Maintain the original tone, emotion, and character personality. "
            "Ensure the translated text is concise enough to be spoken within the original timeframe. "
            "Output ONLY the translated text, no explanations or additional content."
        )
        
        for segment in tqdm(segments, desc="Translating"):
            original_text = segment.get("text", "")
            if not original_text.strip():
                translated_segments.append(segment)
                continue
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this subtitle: '{original_text}'"}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.3
                }
            }
            
            try:
                response = requests.post(self.ollama_url, json=payload)
                response.raise_for_status()
                translated_text = response.json().get("message", {}).get("content", "").strip()
                
                # Clean up any potential artifacts from the model (like quotes)
                if translated_text.startswith("'") and translated_text.endswith("'"):
                    translated_text = translated_text[1:-1]
                if translated_text.startswith('"') and translated_text.endswith('"'):
                    translated_text = translated_text[1:-1]
                
                new_segment = segment.copy()
                new_segment["original_text"] = original_text
                new_segment["text"] = translated_text
                translated_segments.append(new_segment)
                
            except requests.exceptions.RequestException as e:
                print(f"Ollama translation failed: {e}")
                translated_segments.append(segment)
        
        return translated_segments

if __name__ == "__main__":
    # Test stub
    translator = Translator()
    test_segments = [{"text": "Hello, how are you doing today?", "speaker": "SPEAKER_00"}]
    translated = translator.translate_segments(test_segments, target_lang="German")
    print(json.dumps(translated, indent=2))
