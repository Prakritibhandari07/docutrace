import re
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

def retrieve_comparison_evidence(query: str, document_ids: list, n_results: int = 6):
    """Retrieve evidence separately for every selected document."""
    if not document_ids:
        return [], {}

    evidence = []
    evidence_counter = 1
    evidence_counts = {}

    for document_id in document_ids:
        try:
            retrieved_documents = retrieve_documents(
                query=query,
                document_id=document_id,
                n_results=n_results,
            ) or []
        except Exception as exc:
            evidence_counts[document_id] = {"count": 0, "error": str(exc)}
            continue

        document_count = 0
        for document in retrieved_documents:
            if not isinstance(document, dict):
                continue

            text = str(document.get("text") or "").strip()
            if not text:
                continue

            # Use one simple, globally unique ID format everywhere.
            evidence_id = f"E{evidence_counter}"
            evidence.append({
                "id": evidence_id,
                "document": str(document.get("source") or document_id),
                "source": str(document.get("source") or document_id),
                "document_id": str(document_id),
                "page": document.get("page", "Unknown"),
                "text": text,
                "relevance": document.get("relevance"),
                "distance": document.get("distance"),
            })
            evidence_counter += 1
            document_count += 1

        evidence_counts[document_id] = {"count": document_count, "error": None}

    return evidence, evidence_counts


# ============================================================
# BUILD EVIDENCE CONTEXT
# ============================================================

def build_comparison_context(evidence):
    """Format passages and their exact citation IDs for the model."""
    context_parts = []
    for item in evidence:
        context_parts.append(
            f"EVIDENCE ID: [{item['id']}]\n"
            f"DOCUMENT: {item['document']}\n"
            f"PAGE: {item['page']}\n"
            f"PASSAGE:\n{item['text']}"
        )
    return "\n\n".join(context_parts)


# ============================================================
# GENERATE AND VALIDATE COMPARISON
# ============================================================

def generate_comparison(query: str, context: str, evidence=None):
    """Generate a strictly evidence-grounded comparison."""
    evidence = evidence or []

    valid_ids = {
        str(item.get("id", ""))
        for item in evidence
        if item.get("id")
    }

    document_ids = list(dict.fromkeys(
        str(item.get("document_id", ""))
        for item in evidence
        if item.get("document_id")
    ))

    prompt = f"""
You are an evidence-based document comparison assistant.

Your ONLY job is to answer the user's exact question using the supplied
document passages.

IMPORTANT:
The passages may come from different documents. Treat each passage as
belonging ONLY to the document written above that passage.

STRICT RULES:

1. Use ONLY information explicitly stated in the passages.
2. Do NOT use outside knowledge.
3. Do NOT infer missing information.
4. Every factual statement MUST have one or more exact evidence IDs.
5. Evidence IDs must be copied exactly from "EVIDENCE ID".
6. Never invent, modify, shorten, or rename an evidence ID.
7. Never use filenames as citations.
8. Do not attribute a statement to a document unless that document's
   passage actually supports the statement.
9. To say that BOTH documents agree on something, you MUST cite evidence
   from BOTH documents supporting the SAME fact.
10. If only one document supports a fact, describe it as information
    from that document, NOT as an agreement.
11. A difference in topic, level of detail, or focus is NOT a conflict.
12. Call something a conflict only when two documents explicitly make
    incompatible claims about the same fact.
13. Missing information is NOT a conflict.
14. If the evidence does not establish an answer, say:
    "The retrieved evidence is insufficient to establish this."
15. Do not create facts merely to make the comparison balanced.
16. Do not create quotations. Only quote text that appears exactly in
    the supplied passage.
17. Keep the answer focused on the user's question.
18. Do not create a Sources section.
19. Do not name or describe individual OSI or TCP/IP layers unless
    the supplied passages explicitly provide that information.

20. Do not claim that a specific layer in one model corresponds to
    a specific layer in the other model unless the evidence explicitly
    establishes that relationship.

21. Do not claim that both models share a layer or function merely
    because their descriptions seem similar. Require explicit evidence
    from both documents supporting the same claim.

22. Do not repeat the same factual claim across Overview,
    Key Differences, and Areas of Agreement. Give each section
    a distinct purpose.

23. Cite the evidence ID whose passage explicitly supports the
    specific claim. Evidence IDs are not interchangeable, even when
    their passages come from the same document or page.

24. In Evidence Limitations, mention only missing information that
    directly affects the user's question. If no relevant limitation
    exists, write:
    "No material evidence limitations were identified for the question."
    Never introduce unrelated topics just because they appear in
    other retrieved passages.

VERY IMPORTANT CITATION FORMAT:

CITATION REQUIREMENTS — MUST FOLLOW:

Every factual bullet point must end with at least one valid
evidence citation in square brackets.

Correct examples:
- The OSI model has 7 layers, while TCP/IP has 4 layers. [E2]
- Both models share a common application layer. [E1]

Incorrect examples:
- The OSI model has 7 layers.
- Both models share a common application layer (E1).
- The OSI model has 7 layers (E1, E2).

Use ONLY evidence IDs listed below. Copy each ID exactly.
Never output a factual bullet without a citation.
If evidence does not support a claim, omit the claim.
Do not invent citations to satisfy this requirement.

Valid evidence IDs are:
{", ".join(sorted(valid_ids))}

Use citations like:
[E1]
[E2]

Do NOT use:
[OSI and TCPIP.pdf_E1]
[SWITCHING.pdf_E1]
[citation]
[unknown]
or any other citation format.

DOCUMENTS PRESENT:
{", ".join(document_ids)}

OUTPUT FORMAT:

### Overview
State only what the evidence directly establishes about the question.

### Key Differences
Compare the documents only where the retrieved passages provide
supporting evidence.

### Areas of Agreement
Include an item ONLY when relevant evidence from BOTH documents supports
the same claim. Otherwise write:
"No direct agreement is established by the retrieved evidence."

### Potential Conflicts
Include ONLY genuine contradictions supported by passages from both
documents. Otherwise write:
"No direct contradiction is established by the retrieved evidence."

### Evidence Limitations
Mention only missing evidence that directly affects the user's question.
Do not introduce unrelated topics, such as packet switching,
unless the user asks about them.

DOCUMENT EVIDENCE:
{context}

USER QUESTION:
{query}

Now produce the comparison.
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 900,
                },
            },
            timeout=180,
        )

        response.raise_for_status()

        result = response.json().get("response", "").strip()

        if not result:
            return (
                "The model could not generate a comparison "
                "from the retrieved evidence."
            )
        

        # Validate only citation-looking tokens.
        def validate_citation(match):
            citation_id = match.group(1).strip()

            if citation_id in valid_ids:
                return f"[{citation_id}]"

            return "[citation not verified]"

        
        result = re.sub(
            r"\[([A-Za-z0-9_. -]+)\]",
            validate_citation,
            result,
        )

        # Use the passage that explicitly supports both layer counts.
        layer_count_evidence_ids = [
            str(item.get("id", ""))
            for item in evidence
            if item.get("id")
            and re.search(
                r"OSI model.*?\b7\s+(?:architectural\s+)?layers?",
                str(item.get("text", "")),
                re.IGNORECASE | re.DOTALL,
            )
            and re.search(
                r"TCP/IP.*?\b4\s+layers?",
                str(item.get("text", "")),
                re.IGNORECASE | re.DOTALL,
            )
        ]

        if layer_count_evidence_ids:
            expected_id = layer_count_evidence_ids[0]
            corrected_lines = []

            for output_line in result.splitlines():
                has_layer_counts = (
                    re.search(
                        r"\b7\s+(?:architectural\s+)?layers?\b",
                        output_line,
                        re.IGNORECASE,
                    )
                    and re.search(
                        r"\b4\s+layers?\b",
                        output_line,
                        re.IGNORECASE,
                    )
                    and re.search(r"\bOSI\b", output_line, re.IGNORECASE)
                    and re.search(r"\bTCP/IP\b", output_line, re.IGNORECASE)
                )

                if has_layer_counts:
                    output_line = re.sub(
                        r"(?:\s*\[E\d+\])+\s*$",
                        f" [{expected_id}]",
                        output_line,
                    )

                corrected_lines.append(output_line)

            result = "\n".join(corrected_lines)

        # Reject factual statements without verified citations.
        citation_exemptions = {
            "No direct agreement is established by the retrieved evidence.",
            "No direct contradiction is established by the retrieved evidence.",
            "The retrieved evidence is insufficient to establish this.",
        }

        current_section = ""

        for line in result.splitlines():
            line = line.strip()

            if not line:
                continue

            # Track the current output section.
            if line.startswith("### "):
                current_section = line[4:].strip().lower()
                continue

            # Remove Markdown bullet prefixes before validation.
            line = re.sub(r"^[-*•]\s*", "", line)

            # These standard responses describe missing evidence,
            # rather than making ordinary factual claims.
            if line in {
                "No direct agreement is established by the retrieved evidence.",
                "No direct contradiction is established by the retrieved evidence.",
                "The retrieved evidence is insufficient to establish this.",
            }:
                continue

            # Evidence limitations describe what the retrieved
            # passages do not establish.
            if current_section == "evidence limitations":
                continue

            # All other statements must contain a verified citation.
            if (
                "[citation not verified]" in line
                or not re.search(r"\[E\d+\]", line)
            ):
                return (
                    "The comparison could not be verified because "
                    "a statement lacked a valid evidence citation. "
                    "Please retry the comparison."
                )
        return result

    except requests.exceptions.RequestException as exc:
        return (
            "Comparison generation failed. Check whether Ollama is running "
            f"and the model '{MODEL_NAME}' is available. Details: {exc}"
        )

    except (ValueError, TypeError) as exc:
        return f"Ollama returned an unexpected response: {exc}"
# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_documents(query: str, document_ids: list, n_results: int = 6):
    """Compare selected documents only when each supplies usable evidence."""
    if not query or not query.strip():
        return "Please enter a comparison question.", []

    if not document_ids or len(document_ids) < 2:
        return "Please select at least two documents to compare.", []

    document_ids = list(dict.fromkeys(
        str(document_id).strip()
        for document_id in document_ids
        if document_id and str(document_id).strip()
    ))

    if len(document_ids) < 2:
        return "Please select at least two different documents to compare.", []

    evidence, evidence_counts = retrieve_comparison_evidence(
        query=query.strip(),
        document_ids=document_ids,
        n_results=n_results,
    )

    missing_documents = [
        document_id for document_id in document_ids
        if evidence_counts.get(document_id, {}).get("count", 0) == 0
    ]

    if missing_documents:
        details = []
        for document_id in missing_documents:
            status = evidence_counts.get(document_id, {})
            if status.get("error"):
                details.append(f"- {document_id}: retrieval failed ({status['error']})")
            else:
                details.append(f"- {document_id}: no usable passage was retrieved")

        message = (
            "Comparison not generated because evidence was not retrieved "
            "from every selected document.\n\nMissing evidence:\n"
            + "\n".join(details)
            + "\n\nTry a more specific question or check that each document is indexed."
        )
        return message, evidence

    if not evidence:
        return "Insufficient evidence: no usable passages were retrieved.", []

    context = build_comparison_context(evidence)
    comparison = generate_comparison(query.strip(), context, evidence=evidence)
    return comparison, evidence
