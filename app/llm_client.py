import os
import requests
import streamlit as st

def get_config(name, default=None):
    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

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
    Reject empty or malformed model responses.
    """

    provider = get_config("LLM_PROVIDER", "ollama").lower()

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

        data = response.json()
        output = str(data.get("response") or "").strip()

        if not output:
            raise ValueError(
                f"Ollama returned an empty response. "
                f"Response fields: {list(data.keys())}"
            )

        return output

    if provider == "openrouter":
        api_key = get_config("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")

        payload = {
            "model": get_config("OPENROUTER_MODEL", "openrouter/free"),
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

        data = response.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"OpenRouter returned an unexpected response: {data}"
            )

        output = str(content or "").strip()

        if not output:
            raise ValueError(
                f"OpenRouter returned empty message content. "
                f"Response: {data}"
            )

        return output

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

