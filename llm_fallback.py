import os
import google.generativeai as genai

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY")
]

def call_llm_with_fallback(prompt: str) -> str:
    for key in API_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel(model_name="gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Failed with key: {key[:10]}..., Error: {e}")
            continue
    return "⚠️ All LLM API keys failed. Please check usage or network."