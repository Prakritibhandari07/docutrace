import re
from difflib import SequenceMatcher

from app.retriever import retrieve_documents, get_collection
from app.llm import generate_answer

def get_adjacent_page_context(retrieved_documents):
    """
    Fetch the preceding page when a retrieved passage contains
    a likely reference to information introduced earlier.

    Context is kept separate from the original evidence so that
    its document and page citations remain accurate.
    """
    collection = get_collection()

    context_documents = {}
    pronoun_pattern = re.compile(
        r"\b(it|they|this|these|those|such|its|their)\b",
        re.IGNORECASE
    )

    for index, doc in enumerate(retrieved_documents, start=1):
        page = doc.get("page")
        source = doc.get("source")
        text = doc.get("text", "")

        if not source or not isinstance(page, int) or page <= 1:
            continue

        if not pronoun_pattern.search(text):
            continue

        previous_page = page - 1
        key = (source, previous_page)

        if key not in context_documents:
            results = collection.get(
                where={
                    "$and": [
                        {"source": source},
                        {"page": previous_page}
                    ]
                },
                include=["documents", "metadatas"]
            )

            context_documents[key] = {
                "documents": results.get("documents") or [],
                "metadatas": results.get("metadatas") or [],
                "context_for": set()
            }

        context_documents[key]["context_for"].add(f"E{index}")

    return [
        {
            "source": source,
            "page": page,
            "text": text,
            "context_for": sorted(data["context_for"])
        }
        for (source, page), data in context_documents.items()
        for text in data["documents"]
    ]

def normalize_text(text):
    """Normalize text for similarity comparison."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_similarity(text1, text2):
    """Return a simple similarity score between two pieces of text."""
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(None, text1, text2).ratio()


def get_keyword_overlap(claim, evidence_text):
    """
    Calculate how many meaningful words from the claim
    occur in the evidence.
    """
    claim_words = set(normalize_text(claim).split())
    evidence_words = set(normalize_text(evidence_text).split())

    # Ignore very common words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "of", "to", "and", "or", "for",
        "with", "by", "from", "this", "that", "it",
        "as", "at", "be", "can", "which"
    }

    claim_words = {
        word for word in claim_words
        if word not in stop_words and len(word) > 2
    }

    if not claim_words:
        return 0.0

    overlap = claim_words.intersection(evidence_words)

    return len(overlap) / len(claim_words)


def find_supporting_evidence(claim, evidence):
    """
    Find the evidence passages that best support a claim.
    Returns one or two evidence IDs.
    """

    scores = []

    for item in evidence:
        evidence_text = item.get("text", "")

        similarity = get_similarity(
            claim,
            evidence_text
        )

        keyword_score = get_keyword_overlap(
            claim,
            evidence_text
        )

        # Keyword overlap is more useful for document evidence
        score = (
            similarity * 0.35
            + keyword_score * 0.65
        )

        scores.append(
            (
                score,
                item["id"]
            )
        )

    scores.sort(reverse=True)

    if not scores:
        return []

    # Strongest evidence
    best_score, best_id = scores[0]

    citations = []

    # Require reasonable lexical support
    if best_score >= 0.18:
        citations.append(best_id)

    # Add a second source only when it provides meaningful support
    if len(scores) > 1:
        second_score, second_id = scores[1]

        if (
            second_score >= 0.22
            and second_score >= best_score * 0.70
        ):
            citations.append(second_id)

    return citations



def remove_existing_citations(text):
    """
    Remove existing E-style citations so we can attach
    deterministic citations ourselves.
    """
    return re.sub(
        r"\s*\[E\d+\]",
        "",
        text
    ).strip()



def deduplicate_bullets(answer):
    """
    Remove highly similar bullet points.
    """

    lines = answer.splitlines()

    result = []
    previous_claims = []

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        # Only process bullet lines
        if stripped.startswith("-"):
            claim = stripped.lstrip("- ").strip()

            # Remove citations before comparing
            claim = remove_existing_citations(claim)

            duplicate = False

            for previous in previous_claims:
                similarity = get_similarity(
                    claim,
                    previous
                )

                if similarity >= 0.78:
                    duplicate = True
                    break

            if duplicate:
                continue

            previous_claims.append(claim)
            result.append(claim)

        else:
            result.append(stripped)

    return result


def attach_citations(answer, evidence):
    """
    Attach deterministic evidence citations to generated claims.
    """

    valid_ids = {
        item["id"]
        for item in evidence
    }

    lines = deduplicate_bullets(answer)

    final_lines = []

    for line in lines:

        # Remove any citations generated by the LLM
        clean_line = remove_existing_citations(line)

        if not clean_line:
            continue

        # Don't attach citations to headings
        if (
            clean_line.startswith("#")
            or clean_line.lower() in {
                "answer",
                "final answer"
            }
        ):
            final_lines.append(clean_line)
            continue

        citations = find_supporting_evidence(
            clean_line,
            evidence
        )

        # Keep only valid evidence IDs
        citations = [
            citation
            for citation in citations
            if citation in valid_ids
        ]
        context_ids = [
            item["id"]
            for item in evidence
            if item.get("context_for")
            and any(
                context_id in citations
                for context_id in item["context_for"]
            )
        ]

        citations = list(dict.fromkeys(citations + context_ids))
        if citations:
            citation_text = "".join(
                f" [{citation}]"
                for citation in citations
            )

            final_lines.append(
                clean_line + citation_text
            )
        else:
            # If we cannot find supporting evidence,
            # do not pretend the claim is supported.
            final_lines.append(
                clean_line
            )

    return "\n".join(final_lines).strip()


def run_rag(
    query: str,
    document_id: str = None,
    document_ids: list = None
):
    """
    Main RAG pipeline.

    1. Retrieve evidence
    2. Create evidence IDs
    3. Build LLM context
    4. Generate answer
    5. Deduplicate claims
    6. Attach deterministic citations
    7. Return answer + evidence
    """

    # --------------------------------------------------
    # RETRIEVE DOCUMENTS
    # --------------------------------------------------

    retrieved_documents = retrieve_documents(
        query,
        document_id=document_id,
        document_ids=document_ids
    )
    # Filter out message-switching passages when the question
    # specifically asks about packet switching.
    query_lower = query.lower()

    asks_about_packet_switching = (
        "packet switching" in query_lower
        and "message switching" not in query_lower
    )

    if asks_about_packet_switching:
        filtered_documents = []

        for doc in retrieved_documents:
            text_lower = str(doc.get("text") or "").lower()

            is_message_switching_focused = (
                "message switching" in text_lower
                and (
                    "messages of unlimited sizes" in text_lower
                    or "illustrating message switching" in text_lower
                    or "message switching network" in text_lower
                )
            )

            if not is_message_switching_focused:
                filtered_documents.append(doc)

        retrieved_documents = filtered_documents




    # --------------------------------------------------
    # INSUFFICIENT EVIDENCE
    # --------------------------------------------------

    if not retrieved_documents:

        if document_ids:
            selected_names = ", ".join(document_ids)
            message = (
                "I could not find enough relevant information "
                f"in the selected documents: {selected_names}."
            )

        elif document_id:
            message = (
                "I could not find enough relevant information "
                f"in {document_id} to answer this question."
            )

        else:
            message = (
                "I could not find enough relevant information "
                "in your document library to answer this question."
            )

        return (
            "⚠️ Insufficient evidence.\n\n" + message,
            []
        )

    # ADD ADJACENT-PAGE CONTEXT
    context_documents = get_adjacent_page_context(
        retrieved_documents
    )


    # --------------------------------------------------
    # CREATE EVIDENCE IDS
    # --------------------------------------------------

    evidence = []

    for index, doc in enumerate(retrieved_documents, start=1):
        evidence.append({
            "id": f"E{index}",
            "document": doc.get("source", "Unknown document"),
            "page": doc.get("page", "Unknown"),
            "relevance": doc.get("relevance"),
            "distance": doc.get("distance"),
            "text": doc.get("text", ""),
            "context_for": []
        })

    # --------------------------------------------------
    # ADD PREVIOUS-PAGE CONTEXT AS SEPARATE EVIDENCE
    # --------------------------------------------------

    for context_doc in context_documents:
        evidence_id = f"E{len(evidence) + 1}"

        evidence.append({
            "id": evidence_id,
            "document": context_doc["source"],
            "page": context_doc["page"],
            "relevance": None,
            "distance": None,
            "text": context_doc["text"],
            "context_for": context_doc["context_for"]
        })

    # --------------------------------------------------
    # BUILD LLM CONTEXT
    # --------------------------------------------------

    context_parts = []

    for item in evidence:
        context_parts.append(
            f"""
[EVIDENCE ID: {item['id']}]

DOCUMENT: {item['document']}

PAGE: {item['page']}

PASSAGE:
{item['text']}
"""
        )

    context = "\n".join(context_parts)

    # --------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------

    answer = generate_answer(
        query,
        context
    )

    if not isinstance(answer, str):
        answer = str(answer)

    # --------------------------------------------------
    # REMOVE LEGACY CITATION FORMATS
    # --------------------------------------------------

    answer = re.sub(
        r"\[SOURCE\s+\d+(?:\s*,\s*SOURCE\s+\d+)*"
        r"(?:\s*,?\s*Page:\s*\d+)?\]",
        "",
        answer,
        flags=re.IGNORECASE
    )

    answer = re.sub(
        r"\n\s*Sources\s*:.*$",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL
    )

    answer = answer.strip()

    # --------------------------------------------------
    # ATTACH DETERMINISTIC CITATIONS
    # --------------------------------------------------

    final_answer = attach_citations(
        answer,
        evidence
    )

    # --------------------------------------------------
    # FALLBACK
    # --------------------------------------------------

    if not final_answer:

        final_answer = (
            "The provided documents do not contain "
            "enough evidence to answer this question."
        )

    # --------------------------------------------------
    # DEBUG
    # --------------------------------------------------

   
    return final_answer, evidence
