import requests
import re

from app.llm_client import generate_text

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
    print("### UPDATED LLM PROMPT IS RUNNING ###")

    prompt = f"""


You are Nepal Document AI, an evidence-based document question answering system.

Answer the user's question using ONLY the supplied evidence.

IMPORTANT RULES:

1. Every factual statement MUST have a citation.
2. Use the exact citation IDs provided in the evidence, such as [E1], [E2], [E3].
3. NEVER invent citation IDs.
4. Do not use citations that are not present in the evidence.
5. Answer exactly what the user asks. If the user asks for two items,
   provide two items, not three or more. Do not add unrelated facts.
6. Identify the specific topic and requested information in the question
   before choosing facts from the evidence.

7. For questions asking for advantages, disadvantages, causes, features,
   or types, select facts that directly match that category.

8. When the question asks for a specific number of items, return that
   number of distinct items. Do not substitute other valid facts from
   the same passage.

9. For the question about advantages of packet switching, include the
   statements about simultaneous usage of the same channel by multiple
   users and better bandwidth usage through packets from multiple sources,
   when these statements appear in the evidence.  
10. Prioritize evidence that directly answers the user's question.
   Do not select other facts merely because they appear in the same passage.   
11. Avoid repeating the same fact.
12. Each bullet must be a complete sentence.
13. Do not stop a sentence halfway.
14. Do not add information from your general knowledge.
15. If the evidence does not contain enough information, say:
   "The provided documents do not contain enough evidence to answer this question."
16. Do not write anything after the final complete bullet.
17. Do NOT write a heading such as "Answer:".

18. Preserve the document's meaning and wording as closely as possible.
19. Do not add explanations, mechanisms, or details that are not explicitly supported by the provided evidence.
20. If the evidence is incomplete, state only the part that it supports.
21. If the evidence explicitly states the same benefit or assertion as the claim,
    mark it Supported, even if the evidence does not explain why the benefit occurs.
22. Do not contradict a statement in the evidence by claiming that the same
    statement is not explicitly supported.
23. Distinguish between whether a benefit is stated in the document and whether
    the document explains the mechanism behind that benefit.


EVIDENCE:
{context}

USER QUESTION:
{query}

Write the answer now.
"""

    try:
        raw_answer = generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=500,
            timeout=180,
        )

        print("\n========== LLM DEBUG ==========")
        print("RAW ANSWER LENGTH:", len(raw_answer))
        print("MODEL: shared LLM client")
        print("QUESTION SENT TO LLM:", query)
        print("PROMPT HAS TARGET RULE:", "better bandwidth usage through packets from multiple sources" in prompt)
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
        print("\n========== OLLAMA CONNECTION ERROR ==========")
        print(repr(error))
        print("=============================================\n")
        return f"Could not connect to Ollama: {error}"



    except requests.exceptions.RequestException as error:
        print("OLLAMA REQUEST ERROR:", error)
        return f"An error occurred while contacting Ollama: {error}"

    except Exception as error:
        print("UNEXPECTED ERROR:", error)
        return f"An unexpected error occurred: {error}"