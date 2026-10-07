import requests
import re


# ================================================
# Ollama configuration
# ================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "qwen2.5:3b"


# ================================================
# Ask Ollama
# ================================================

def ask_ollama(
    prompt: str,
    timeout: int = 120
) -> str:

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 400
            }
        },
        timeout=timeout
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        ""
    ).strip()


# ================================================
# Create manageable document sections
# ================================================

def create_summary_sections(
    pages: list,
    max_chars: int = 3500
) -> list:

    sections = []

    current_section = ""

    current_pages = []


    for page in pages:

        page_number = page.get(
            "page",
            "Unknown"
        )

        text = page.get(
            "text",
            ""
        )

        if not text or not text.strip():
            continue


        page_text = (
            f"\nPAGE {page_number}\n\n"
            f"{text.strip()}\n"
        )


        if (
            len(current_section)
            + len(page_text)
            <= max_chars
        ):

            current_section += page_text

            current_pages.append(
                page_number
            )

        else:

            if current_section.strip():

                sections.append(
                    {
                        "text": current_section,
                        "pages": current_pages
                    }
                )


            current_section = page_text

            current_pages = [
                page_number
            ]


    if current_section.strip():

        sections.append(
            {
                "text": current_section,
                "pages": current_pages
            }
        )


    return sections


# ================================================
# Summarize one section
# ================================================

def summarize_section(
    text: str
) -> str:

    prompt = f"""
You are summarizing part of a document.

Use ONLY the information in the text below.

Do not use outside knowledge.
Do not invent information.

Identify the important topics, concepts, facts,
definitions, models, methods, or explanations.

Write a concise summary.

DOCUMENT TEXT:

{text}

SUMMARY:
"""

    return ask_ollama(
        prompt,
        timeout=120
    )


# ================================================
# Generate document summary
# ================================================

def generate_document_summary(
    document_name: str,
    pages: list
) -> str:

    if not pages:

        return (
            "The document does not contain enough "
            "readable text to generate a summary."
        )


    sections = create_summary_sections(
        pages,
        max_chars=3500
    )


    if not sections:

        return (
            "The document does not contain enough "
            "readable text to generate a summary."
        )


    section_summaries = []


    for index, section in enumerate(
        sections,
        start=1
    ):

        summary = summarize_section(
            section["text"]
        )


        if summary:

            section_summaries.append(
                {
                    "section": index,
                    "summary": summary
                }
            )


    if not section_summaries:

        return (
            "Unable to generate a summary from "
            "the document."
        )


    combined_information = "\n\n".join(
        [
            (
                f"SECTION {item['section']}\n"
                f"{item['summary']}"
            )
            for item in section_summaries
        ]
    )


    final_prompt = f"""
You are a document intelligence assistant.

Create a final summary of the document using ONLY
the section summaries provided below.

Do not use outside knowledge.

Do not invent information.

Document name:

{document_name}

==================================================
SECTION SUMMARIES
==================================================

{combined_information}

==================================================
TASK
==================================================

Create a concise and well-structured final summary.

Use exactly this structure:

### Overview

Write a short paragraph explaining what the document
is mainly about.

### Key Topics

- Topic
- Topic
- Topic

### Important Concepts

- Concept
- Concept
- Concept

### Main Takeaways

1. Important point
2. Important point
3. Important point

==================================================
RULES
==================================================

1. Use ONLY information from the section summaries.

2. Do not add outside knowledge.

3. Do not invent facts.

4. Combine repeated topics naturally.

5. Preserve technical terminology.

6. Do not mention section numbers.

7. Do not include page numbers.

8. Do not create a Sources section.

9. Do not say "Based on my analysis".

10. Start directly with "### Overview".

FINAL SUMMARY:
"""

    final_summary = ask_ollama(
        final_prompt,
        timeout=120
    )


    if not final_summary:

        return (
            "The document was partially analyzed, "
            "but a final summary could not be generated."
        )


    return final_summary


# ================================================
# Extract key topics and page locations
# ================================================

def extract_key_topics(
    pages: list
) -> list:

    if not pages:
        return []


    sections = create_summary_sections(
        pages,
        max_chars=3500
    )


    if not sections:
        return []


    section_information = []


    for index, section in enumerate(
        sections,
        start=1
    ):

        page_numbers = section.get(
            "pages",
            []
        )

        text = section.get(
            "text",
            ""
        )


        section_information.append(
            f"""
SECTION {index}

PAGES: {", ".join(map(str, page_numbers))}

TEXT:

{text}
"""
        )


    combined_information = "\n".join(
        section_information
    )


    prompt = f"""
You are a document intelligence assistant.

Identify the main topics discussed in the document.

Use ONLY the document content provided below.

Do not use outside knowledge.

Do not invent topics.

For each topic, provide the page numbers where the
topic is discussed.

Return ONLY lines in this format:

TOPIC | PAGE: 1, 2, 3

Do not add numbering.

Do not add explanations.

Do not add markdown.

Do not add a heading.

Do not include minor details.

Return approximately 3 to 8 important topics.

==================================================
DOCUMENT CONTENT
==================================================

{combined_information}

==================================================
KEY TOPICS
==================================================
"""

    result = ask_ollama(
        prompt,
        timeout=120
    )


    if not result:
        return []


    topics = []


    for line in result.splitlines():

        line = line.strip()


        if not line:
            continue


        # ----------------------------------------
        # Remove accidental bullets / numbering
        # ----------------------------------------

        line = re.sub(
            r"^[\-\*\d\.\)\s]+",
            "",
            line
        ).strip()


        # ----------------------------------------
        # Parse:
        # Topic | PAGE: 1, 2
        # ----------------------------------------

        match = re.match(
            r"(.+?)\s*\|\s*PAGE(?:S)?\s*:\s*(.+)",
            line,
            flags=re.IGNORECASE
        )


        if not match:
            continue


        topic = match.group(
            1
        ).strip()


        page_text = match.group(
            2
        ).strip()


        # ----------------------------------------
        # Extract page numbers
        # ----------------------------------------

        page_numbers = re.findall(
            r"\d+",
            page_text
        )


        if not topic or not page_numbers:
            continue


        # ----------------------------------------
        # Remove duplicate pages
        # ----------------------------------------

        unique_pages = []

        for page in page_numbers:

            if page not in unique_pages:

                unique_pages.append(
                    page
                )


        topics.append(
            {
                "topic": topic,
                "pages": unique_pages
            }
        )


    return topics