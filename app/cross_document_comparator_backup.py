import requests

from app.retriever import retrieve_documents


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# ============================================================
# RETRIEVE EVIDENCE FROM EACH DOCUMENT
# ============================================================

def retrieve_comparison_evidence(
    query: str,
    document_ids: list,
    n_results: int = 6
):
    """
    Retrieve evidence separately from each selected document.

    Each evidence passage receives a unique evidence ID:
    E1, E2, E3, etc.
    """

    if not document_ids:
        return []

    evidence = []
    evidence_counter = 1

    for document_id in document_ids:

        retrieved_documents = retrieve_documents(
            query=query,
            document_id=document_id,
            n_results=n_results
        )

        for document in retrieved_documents:

            evidence_id = f"E{evidence_counter}"
            evidence_counter += 1

            evidence.append({
                "id": evidence_id,
                "document": document.get(
                    "source",
                    document_id
                ),
                "page": document.get(
                    "page",
                    "Unknown"
                ),
                "text": document.get(
                    "text",
                    ""
                ),
                "relevance": document.get(
                    "relevance",
                    None
                ),
                "distance": document.get(
                    "distance",
                    None
                )
            })

    return evidence


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_comparison_context(evidence):
    """
    Format retrieved evidence for the comparison model.
    """

    context_parts = []

    for item in evidence:

        context_parts.append(
            f"""
==================================================
EVIDENCE {item['id']}
==================================================

DOCUMENT: {item['document']}
PAGE: {item['page']}

PASSAGE:
{item['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE CROSS-DOCUMENT COMPARISON
# ============================================================

def generate_comparison(
    query: str,
    context: str
):
    """
    Generate a comparison using only retrieved evidence.
    """

    prompt = f"""
You are a cross-document evidence comparison assistant.

Your task is to compare information across the selected
documents using ONLY the evidence provided below.

==================================================
STRICT RULES
==================================================

1. Use only information present in the evidence.
2. Do not use outside knowledge.
3. Do not invent facts or fill missing information.
4. Compare documents only where relevant evidence exists.
5. Do not assume that missing information means disagreement.
6. Distinguish between differences and actual contradictions.
7. If evidence is insufficient, clearly say so.
8. Keep the comparison clear and concise.
9. Do not invent document names, page numbers, or evidence IDs.
10. Do not create a separate Sources section.

==================================================
CITATION RULES
==================================================

Each evidence passage has an ID such as E1, E2, or E3.

Whenever you make a factual statement, cite the evidence
that supports it.

Example:
The first document describes seven layers. [E1]

The second document describes four layers. [E3]

Use only evidence IDs provided below.

==================================================
COMPARISON STRUCTURE
==================================================

Organize the answer using these sections:

### Overview
Briefly describe what the evidence from the documents covers.

### Key Differences
Explain the relevant differences between the documents.
Use a table when it makes the comparison clearer.

### Areas of Agreement
Identify information that the documents explicitly share.

### Potential Conflicts
Identify statements that appear to contradict each other.

Do not label a difference as a contradiction unless the
evidence supports that conclusion.

If a section has no relevant evidence, state that clearly.

==================================================
DOCUMENT EVIDENCE
==================================================

{context}

==================================================
USER QUESTION
==================================================

{query}

==================================================
COMPARISON
==================================================

Compare the documents using only the evidence above.
Include evidence citations with each factual statement.
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1000
                }
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json().get(
            "response",
            ""
        ).strip()

        if not result:
            return (
                "The model could not generate a comparison "
                "from the provided evidence."
            )

        return result

    except requests.exceptions.RequestException as e:

        return (
            "Comparison generation failed. "
            f"Check whether Ollama is running. Details: {e}"
        )


# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_documents(
    query: str,
    document_ids: list,
    n_results: int = 6
):
    """
    Main function for cross-document comparison.

    Returns:
        comparison: Generated comparison text.
        evidence: Retrieved evidence with IDs and metadata.
    """

    if not query or not query.strip():

        return (
            "Please enter a comparison question.",
            []
        )

    if not document_ids or len(document_ids) < 2:

        return (
            "Please select at least two documents "
            "to compare.",
            []
        )

    # Step 1: Retrieve evidence from each document.
    evidence = retrieve_comparison_evidence(
        query=query,
        document_ids=document_ids,
        n_results=n_results
    )

    # Step 2: Handle insufficient evidence.
    if not evidence:

        return (
            "Insufficient evidence. No relevant information "
            "was found in the selected documents.",
            []
        )

    # Step 3: Build the context.
    context = build_comparison_context(
        evidence
    )

    # Step 4: Generate the comparison.
    comparison = generate_comparison(
        query=query,
        context=context
    )

    return comparison, evidence