import re
import requests

from app.retriever import (
    retrieve_documents,
    extract_query_keywords,
    keyword_overlap,
)


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
    n_results: int = 6,
):
    """Retrieve and rank evidence separately for every selected document."""
    print(f"[Comparison Debug] document_ids received: {document_ids!r}")
    """Retrieve and rank evidence separately for every selected document."""
    if not document_ids:
        return [], {}

    evidence = []
    evidence_counts = {}
    query_keywords = extract_query_keywords(query)

    for document_id in document_ids:
        try:
            print(
                f"[Comparison Debug] Selected document ID: "
                f"{document_id!r}"
            )

            retrieved_documents = retrieve_documents(
                query=query,
                document_id=document_id,
                n_results=max(n_results, 12),
                ) or []

            print(
                f"[Comparison Debug] Requested document: {document_id!r}"
            )
            print(
                f"[Comparison Debug] Retrieved sources: "
                f"{[doc.get('source') for doc in retrieved_documents]}"
            )
            print(
                f"[Comparison Debug] Retrieved count: "
                f"{len(retrieved_documents)}"
            )



            candidates = []
            for doc in retrieved_documents:
                if not isinstance(doc, dict):
                    continue

                text = str(doc.get("text") or "").strip()
                if not text:
                    continue


                text_lower = text.lower()
                query_lower = query.lower()


                # Filter passages for advantage questions.
                asks_advantages = (
                    "advantage" in query_lower or "benefit" in query_lower
                )

                if asks_advantages:
                    has_advantage_heading = any(
                        word in text_lower
                        for word in ("advantages", "advantage", "benefits", "benefit")
                    )

                    is_packet_switching_context = (
                        "packet switching" in text_lower
                        and "message switching" not in text_lower
                        and "disadvantages" not in text_lower
                        and "disadvantage" not in text_lower
                    )

                    if not (has_advantage_heading or is_packet_switching_context):
                        print(
                            f"[Comparison Debug] Skipped irrelevant passage: "
                            f"{document_id}, page {doc.get('page', 'Unknown')}"
                        )
                        continue

                # Filter passages that focus on message switching.
                is_message_switching_focused = (
                    "message switching" in text_lower
                    and (
                        "messages of unlimited sizes" in text_lower
                        or "illustrating message switching" in text_lower
                        or "message switching network" in text_lower
                    )
                )

                if is_message_switching_focused:
                    print(
                        f"[Comparison Debug] Skipped message-switching passage: "
                        f"{document_id}, page {doc.get('page', 'Unknown')}"
                    )
                    continue

                # Filter passages for disadvantage questions.
                asks_disadvantages = (
                    "disadvantage" in query_lower
                    or "limitation" in query_lower
                    or "drawback" in query_lower
                )

                if asks_disadvantages:
                    has_disadvantage_heading = any(
                        word in text_lower
                        for word in (
                            "disadvantages",
                            "disadvantage",
                            "limitations",
                            "limitation",
                            "drawbacks",
                            "drawback",
                        )
                    )

                    if not has_disadvantage_heading:
                        print(
                            f"[Comparison Debug] Skipped passage without "
                            f"disadvantages: {document_id}, "
                            f"page {doc.get('page', 'Unknown')}"
                        )
                        continue

                overlap = keyword_overlap(query_keywords, text)
                relevance = float(doc.get("relevance") or 0)

                print(
                    f"[Comparison Debug] {document_id}: "
                    f"relevance={relevance}, "
                    f"keyword_overlap={overlap}"
                )

                if relevance >= 0.50 and overlap >= 2:
                    candidates.append(doc)
                    print(
                        f"[Comparison Debug] ACCEPTED: {document_id}, "
                        f"page {doc.get('page', 'Unknown')}"
                    )
                else:
                    print(
                        f"[Comparison Debug] REJECTED: {document_id}, "
                        f"page {doc.get('page', 'Unknown')}, "
                        f"relevance={relevance}, overlap={overlap}"
                    )


            # Rank each document's candidates by relevance.
            candidates.sort(
                key=lambda item: float(item.get("relevance") or 0),
                reverse=True,
            )

            # Retain up to three passages from each document.
            selected_documents = candidates[:3]
            document_count = 0

            for document in selected_documents:
                text = str(document.get("text") or "").strip()
                if not text:
                    continue

                evidence_id = f"E{len(evidence) + 1}"
                source = str(document.get("source") or document_id)

                evidence.append({
                    "id": evidence_id,
                    "document": source,
                    "source": source,
                    "document_id": str(document_id),
                    "page": document.get("page", "Unknown"),
                    "text": text,
                    "relevance": document.get("relevance"),
                    "distance": document.get("distance"),
                })

                document_count += 1

            evidence_counts[document_id] = {
                "count": document_count,
                "error": None,
            }

            print(
                f"[Comparison Debug] {document_id}: "
                f"{document_count} relevant chunks retained"
            )

        except Exception as exc:
            evidence_counts[document_id] = {
                "count": 0,
                "error": str(exc),
            }
            print(
                f"[Comparison Debug] {document_id}: ERROR: {exc}"
            )

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
    """Generate an evidence-grounded answer and verify citation IDs."""
    evidence = evidence or []

    valid_ids = {
        str(item.get("id", "")).strip()
        for item in evidence
        if item.get("id")
    }

    if not valid_ids:
        return "Insufficient evidence: no citation sources are available."

    asks_two_sentences = bool(
        re.search(r"\btwo sentences\b|\b2 sentences\b", query, re.I)
    )
    asks_one_difference = bool(
        re.search(r"\bone difference\b|\bsingle difference\b", query, re.I)
    )

    if asks_two_sentences:
        format_rule = "Answer in exactly TWO concise sentences."
        num_predict = 180
    elif asks_one_difference:
        format_rule = "Give exactly ONE difference in one or two sentences."
        num_predict = 150
    else:
        format_rule = "Give a concise answer with 2-4 key points."
        num_predict = 300

    prompt = f"""
You are DocuTrace, a document comparison assistant.

Answer the user's question using ONLY the retrieved evidence.

IMPORTANT CONCEPT DISTINCTION:
- Treat "circuit switching" and "virtual circuit switching" as
  DIFFERENT networking techniques.
- Do not use evidence about virtual circuit switching to explain
  traditional circuit switching.
- When comparing two concepts, identify evidence that directly
  describes each concept before stating their differences.
- If the retrieved evidence does not adequately explain one concept,
  explicitly say that the evidence is insufficient for that part.
- Never combine facts about different techniques as though they
  describe the same technique.

MANDATORY CITATION RULES:
- Every factual sentence must end with one or more valid evidence citations.
- Use the exact format [E1], [E2], and so on.
- Cite only evidence IDs shown in the retrieved evidence.
- Choose IDs whose passages directly support the sentence.
- Never invent a citation or attach unrelated citations.
- If evidence is insufficient, say so and cite the evidence supporting that limitation.
- Do not add an introduction or conclusion.

Example:
Packet switching divides a message into individually routed packets [E3].
Each packet contains a header and a payload [E3].

{format_rule}

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

Return the answer with citations included in the text.
"""

    def call_ollama(current_prompt, limit):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": current_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": limit,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def extract_valid_citations(text):
        found = re.findall(r"\[(E\d+)\]", text)
        return [citation for citation in found if citation in valid_ids]

    def every_sentence_has_valid_citation(text):
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
            if sentence.strip()
        ]

        return bool(sentences) and all(
            extract_valid_citations(sentence)
            for sentence in sentences
        )


    try:
        result = call_ollama(prompt, num_predict)

        if not result:
            return (
                "The model could not generate an answer "
                "from the retrieved evidence."
            )

        # If the model ignored citations, retry with a stricter instruction.
        if not every_sentence_has_valid_citation(result):
            retry_prompt = f"""
Answer this question using only the evidence below:
{query}

Evidence:
{context}

Available citation IDs: {", ".join(f"[{item}]" for item in sorted(
    valid_ids, key=lambda value: int(value[1:])
))}

Write 2-4 concise factual sentences. Each factual sentence MUST end
with a citation copied exactly from the available IDs. Select citations
that directly support each sentence. Do not output uncited factual claims.
Output only the answer.
"""
            retry_result = call_ollama(retry_prompt, num_predict)

            if retry_result and every_sentence_has_valid_citation(retry_result):
                result = retry_result

        # Remove malformed or unknown citations, retaining valid ones.
        def validate_citation(match):
            citation_id = match.group(1)
            return f"[{citation_id}]" if citation_id in valid_ids else ""

        result = re.sub(
            r"\[(E\d+)\]",
            validate_citation,
            result,
        )

        result = re.sub(r"\n{3,}", "\n\n", result).strip()

        # Do not silently present a completely uncited answer as verified.
        if not every_sentence_has_valid_citation(result):
            return (
                "The model generated an answer with missing or invalid "
                "sentence-level citations. Please retry the question; "
                "the retrieved evidence is still available below."
            )

        return result

    except requests.exceptions.RequestException as exc:
        return (
            "Comparison generation failed. Check whether Ollama is running "
            f"and the model '{MODEL_NAME}' is available. Details: {exc}"
        )
    except (ValueError, TypeError) as exc:
        return f"Could not process the comparison response: {exc}"


def detect_contradictions(query: str, evidence: list):
    """Identify evidence-backed contradictions across documents."""
    if len(evidence) < 2:
        return ""

    cross_document_evidence = [
        item for item in evidence
        if item.get("id")
        and item.get("text")
        and item.get("document_id")
    ]

    if len({
        item["document_id"] for item in cross_document_evidence
    }) < 2:
        return ""

    context = build_comparison_context(cross_document_evidence)

    valid_ids = {
        item["id"] for item in cross_document_evidence
    }

    prompt = f"""
You are DocuTrace's contradiction detection assistant.

Question:
{query}

Evidence:
{context}

Identify only genuine factual contradictions between different documents.
A contradiction means two sources make incompatible claims about the
same subject and attribute.

Rules:
- Do not confuse different perspectives or complementary facts with
  contradictions.
- Do not invent facts.
- Every finding must cite evidence from BOTH documents.
- Use only the supplied evidence IDs.
- Include document names and page numbers.
- If no clear contradiction exists, state:
  No clear contradictions were identified.
- Return concise Markdown only.

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
                    "num_predict": 500,
                },
            },
            timeout=180,
        )
        response.raise_for_status()

        result = response.json().get("response", "").strip()

        if not result:
            return ""

        cited_ids = set(re.findall(r"\[(E\d+)\]", result))

        # Require valid citations from at least two documents.
        cited_evidence = [
            item for item in cross_document_evidence
            if item["id"] in cited_ids
        ]

        cited_documents = {
            item["document_id"] for item in cited_evidence
        }

        # A no-contradictions result does not require citations.
        if re.fullmatch(
            r"\s*no clear contradictions were identified\.?\s*",
            result,
            re.I,
        ):
            return result

        # Other findings must cite valid evidence from both documents.
        if (
            not cited_ids
            or not cited_ids.issubset(valid_ids)
            or len(cited_documents) < 2
        ):
            return ""

        return result
    except (
        requests.exceptions.RequestException,
        ValueError,
        TypeError,
    ):
        return ""



# ============================================================
# MAIN COMPARISON FUNCTION
# ============================================================

def compare_documents(query: str, document_ids: list, n_results: int = 6):
    """Compare documents using relevant, document-specific evidence."""
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

    if not evidence:
        return "Insufficient evidence: no usable passages were retrieved.", []

    # Keep evidence balanced across documents and limit prompt size.
    balanced_evidence = []
    for document_id in document_ids:
        print(f"[Comparison Debug] Processing document: {document_id!r}")
        document_evidence = [
            item for item in evidence
            if item.get("document_id") == document_id
        ]
        balanced_evidence.extend(document_evidence[:3])


    evidence = balanced_evidence

    # Identify documents that have no retrieved evidence.
    missing_documents = [
        document_id
        for document_id in document_ids
        if not any(
            str(item.get("document_id", "")).strip() == str(document_id).strip()
            for item in evidence
        )
    ]

    # Build context and generate the answer from this exact evidence list.
    context = build_comparison_context(evidence)

    comparison = generate_comparison(
        query.strip(),
        context,
        evidence=evidence,
    )


    # Detect contradictions across the selected documents.
    contradictions = detect_contradictions(
        query.strip(),
        evidence,
    )

    if contradictions:
        comparison += (
            "\n\n### Potential Contradictions\n"
            + contradictions
        )


    # Report documents without evidence.
    if missing_documents:
        comparison += (
            "\n\n### Evidence Limitations\n"
            "No relevant evidence was retrieved from: "
            + ", ".join(missing_documents)
            + ". Claims about those documents cannot be verified."
        )

    return comparison, evidence
