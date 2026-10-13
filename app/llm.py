import requests
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def clean_answer(answer):
    """
    Remove incomplete final bullets/sentences caused by the model
    stopping in the middle of generation.
    """

    answer = answer.strip()

    if not answer:
        return answer

    # Split into lines
    lines = answer.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        cleaned_lines.append(line)

    if not cleaned_lines:
        return answer

    # Check the last line for an obviously incomplete ending.
    last = cleaned_lines[-1]

    incomplete_endings = [
        "networking re",
        "networking res",
        "networking reso",
        "networking resour",
        "networking resource",
        "networking resources",
        "the networking",
        "the network",
        "and",
        "or",
        "but",
        "because",
        "which",
        "where",
        "that",
        "with",
        "from",
        "to",
        "in",
        "of",
    ]

    lower_last = last.lower().rstrip(" .,;:")

    # Remove obviously incomplete final bullet
    for ending in incomplete_endings:
        if lower_last.endswith(ending):
            cleaned_lines.pop()
            break

    # Remove a final line that ends in a suspicious unfinished word.
    if cleaned_lines:
        last = cleaned_lines[-1].strip()

        # If it starts as a bullet but has no reasonable ending,
        # remove it only when it is very short.
        if last.startswith("-") and len(last.split()) <= 5:
            cleaned_lines.pop()

    return "\n".join(cleaned_lines).strip()


def generate_answer(query, context):
    """
    Generate an evidence-backed answer using Ollama.
    """

    prompt = f"""
You are Nepal Document AI, an evidence-based document question answering system.

Answer the user's question using ONLY the supplied evidence.

IMPORTANT RULES:

1. Every factual statement MUST have a citation.
2. Use the exact citation IDs provided in the evidence, such as [E1], [E2], [E3].
3. NEVER invent citation IDs.
4. Do not use citations that are not present in the evidence.
5. Give a concise answer with 3 to 5 bullet points.
6. Avoid repeating the same fact.
7. Each bullet must be a complete sentence.
8. Do not stop a sentence halfway.
9. Do not add information from your general knowledge.
10. If the evidence does not contain enough information, say:
   "The provided documents do not contain enough evidence to answer this question."
11. Do not write anything after the final complete bullet.
12. Do NOT write a heading such as "Answer:".

EVIDENCE:
{context}

USER QUESTION:
{query}

Write the answer now.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 500,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
        )

        response.raise_for_status()

        result = response.json()

        raw_answer = result.get("response", "")

        print("\n========== OLLAMA DEBUG ==========")
        print("MODEL:", MODEL_NAME)
        print("RAW ANSWER LENGTH:", len(raw_answer))
        print("DONE REASON:", result.get("done_reason"))
        print("EVALUATED TOKENS:", result.get("eval_count"))
        print("RAW ANSWER:", repr(raw_answer))
        print("RAW ANSWER END:", repr(raw_answer[-500:]))
        print("==================================\n")

        answer = clean_answer(raw_answer)

        if not answer:
            return (
                "The provided documents do not contain enough evidence "
                "to answer this question."
            )

        return answer

    except requests.exceptions.Timeout:
        return "The model took too long to respond. Please try again."

    except requests.exceptions.ConnectionError as error:
        print("OLLAMA CONNECTION ERROR:", repr(error))
        return (
            "Could not connect to Ollama. "
            "Check the terminal for the connection error."
        )

    except requests.exceptions.RequestException as error:
        print("OLLAMA REQUEST ERROR:", error)
        return f"An error occurred while contacting Ollama: {error}"

    except Exception as error:
        print("UNEXPECTED ERROR:", error)
        return f"An unexpected error occurred: {error}"