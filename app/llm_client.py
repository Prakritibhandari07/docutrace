import os
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def generate_text(
    prompt,
    temperature=0,
    max_tokens=500,
    timeout=180,
):
    """
    Generate text using local Ollama or hosted OpenRouter.
    Local Ollama remains the default.
    """

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        return response.json().get("response", "").strip()

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        payload = {
            "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://docutrace.streamlit.app",
                "X-Title": "DocuTrace",
            },
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]
        return (content or "").strip()

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
