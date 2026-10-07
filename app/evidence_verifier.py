import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """Normalize text for comparison."""
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# CLAIM EXTRACTION
# ============================================================

def split_into_claims(text):
    """
    Split a text segment into individual sentences.

    Each sentence becomes a separate claim for verification.
    """
    if not text:
        return []

    # Remove Markdown headings.
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)

    # Remove common Markdown bullet markers.
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)

    # Remove numbered-list markers.
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)

    # Remove standalone Markdown separators.
    text = re.sub(r"(?m)^\s*[-_=]{3,}\s*$", "", text)

    # Split at sentence-ending punctuation.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())

    claims = []

    for part in parts:
        claim = part.strip()

        # Remove leftover Markdown formatting.
        claim = re.sub(r"^[`*_]+|[`*_]+$", "", claim).strip()

        if not claim:
            continue

        # Ignore headings and fragments without meaningful words.
        if not re.search(r"[A-Za-z0-9]", claim):
            continue

        claims.append(claim)

    return claims


def extract_claims(answer: str):
    """
    Extract individual claims and their associated evidence IDs.

    Supports:
        Claim one. [E1]
        Claim two. [E2]
        Claim one. Claim two. [E1]
        Claim one. [E1][E2]

    If several sentences precede one citation, the citation is
    associated with each sentence in that segment.

    Uncited sentences are also extracted, but receive no evidence
    IDs. They will therefore be marked Unsupported.
    """
    if not answer or not answer.strip():
        return []

    citation_pattern = re.compile(
        r"(?:\[\s*E\d+\s*\]\s*)+",
        re.IGNORECASE
    )

    claims = []
    matches = list(citation_pattern.finditer(answer))
    previous_end = 0

    for match in matches:
        # Text since the previous citation.
        segment = answer[previous_end:match.start()]

        evidence_ids = re.findall(
            r"\[\s*(E\d+)\s*\]",
            match.group(),
            re.IGNORECASE
        )

        evidence_ids = list(
            dict.fromkeys(
                evidence_id.upper()
                for evidence_id in evidence_ids
            )
        )

        segment_claims = split_into_claims(segment)

        for claim in segment_claims:
            claims.append({
                "claim": claim,
                "evidence_ids": evidence_ids.copy()
            })

        previous_end = match.end()

    # Handle any text after the final citation.
    # Such claims have no citation and cannot be supported by
    # the verifier's citation-based evidence lookup.
    remaining_text = answer[previous_end:]

    for claim in split_into_claims(remaining_text):
        claims.append({
            "claim": claim,
            "evidence_ids": []
        })

    return claims


# ============================================================
# EVIDENCE LOOKUP
# ============================================================

def get_claim_evidence(evidence_ids, evidence):
    """Return only evidence explicitly cited by a claim."""
    evidence_lookup = {
        item.get("id", "").upper(): item
        for item in evidence
        if item.get("id")
    }

    selected = []

    for evidence_id in evidence_ids:
        item = evidence_lookup.get(evidence_id.upper())

        if item:
            selected.append(item)

    return selected


# ============================================================
# DIRECT SUPPORT CHECK
# ============================================================

def direct_support_check(claim, evidence_text):
    """
    Calculate lexical overlap as a preliminary signal.

    Lexical overlap alone does not prove that a claim is supported.
    This function is retained for diagnostics and testing.
    """
    claim_words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        normalize_text(claim)
    )

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "in", "of", "to", "and", "for", "on", "that",
        "this", "with", "from", "it", "its", "be",
        "as", "by", "which", "where", "when", "does",
        "do", "has", "have", "had", "can", "will",
        "between", "before", "after", "during"
    }

    meaningful_words = [
        word for word in claim_words
        if word not in stop_words and len(word) > 2
    ]

    if not meaningful_words:
        return None

    evidence_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            normalize_text(evidence_text)
        )
    )

    matched_words = [
        word for word in meaningful_words
        if word in evidence_words
    ]

    overlap_ratio = len(matched_words) / len(meaningful_words)

    if overlap_ratio >= 0.80:
        return "Supported"

    if overlap_ratio >= 0.50:
        return "Partially Supported"

    return "Unsupported"


# ============================================================
# LLM VERIFICATION
# ============================================================

def llm_verify_claim(claim, evidence_items):
    """
    Ask the language model to verify one claim against its
    explicitly cited evidence.
    """
    evidence_parts = []

    for item in evidence_items:
        evidence_parts.append(
            f"""
[{item.get('id', 'UNKNOWN')}]
Document: {item.get('document', 'Unknown document')}
Page: {item.get('page', 'Unknown page')}

PASSAGE:
{item.get('text', '')}
"""
        )

    context = "\n".join(evidence_parts)

    prompt = f"""
You are a strict document evidence verification system.

Verify ONE claim using ONLY the supplied evidence.

CLAIM:
{claim}

DOCUMENT EVIDENCE:
{context}

RULES:
1. Use only the supplied document evidence.
2. Do not use outside knowledge.
3. Check every important part of the claim.
4. Textual similarity alone does not prove support.
5. Check whether the evidence supports the claim's meaning.
6. Pay attention to negation, quantities, dates, and conditions.
7. Do not infer details that the evidence does not establish.
8. Use Supported only when all important parts are supported.
9. Use Partially Supported when some parts are supported but
   other important parts are missing or unverified.
10. Use Unsupported when the evidence does not establish
    the claim.
11. Explain briefly why the evidence supports or fails to
    support the claim.
12. Do not mention unrelated facts.

Return exactly this format:

STATUS: Supported
REASON: Explain how the evidence supports the claim.

Choose only one status:
Supported
Partially Supported
Unsupported
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
                    "num_predict": 160
                }
            },
            timeout=120
        )

        response.raise_for_status()

        output = response.json().get("response", "").strip()

    except Exception as exc:
        return {
            "status": "Unsupported",
            "reason": f"Verification failed: {exc}"
        }

    status_match = re.search(
        r"STATUS\s*:\s*(Supported|Partially Supported|Unsupported)\b",
        output,
        re.IGNORECASE
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        output,
        re.IGNORECASE | re.DOTALL
    )

    if status_match:
        raw_status = status_match.group(1).lower()

        if raw_status == "supported":
            status = "Supported"
        elif raw_status == "partially supported":
            status = "Partially Supported"
        else:
            status = "Unsupported"
    else:
        status = "Unsupported"

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "The model did not provide a valid verification reason."
    )

    return {
        "status": status,
        "reason": reason
    }


# ============================================================
# VERIFY ONE CLAIM
# ============================================================

def verify_claim(claim, evidence_ids, evidence):
    """Verify a single claim against its cited evidence."""

    if not evidence_ids:
        return {
            "status": "Unsupported",
            "reason": "This claim has no evidence citation."
        }

    claim_evidence = get_claim_evidence(
        evidence_ids,
        evidence
    )

    if not claim_evidence:
        return {
            "status": "Unsupported",
            "reason": (
                "No matching evidence was found for the "
                "cited evidence IDs."
            )
        }

    # Do not automatically mark a claim Supported based only
    # on lexical overlap. Ask the LLM to check its meaning.
    return llm_verify_claim(
        claim,
        claim_evidence
    )


# ============================================================
# VERIFY COMPLETE ANSWER
# ============================================================

def verify_answer(answer, evidence):
    """
    Verify every extracted claim independently.

    The returned list preserves the existing result structure
    expected by the Streamlit interface.
    """
    claims = extract_claims(answer)

    results = []

    for claim_data in claims:
        claim = claim_data["claim"]
        evidence_ids = claim_data["evidence_ids"]

        verification = verify_claim(
            claim,
            evidence_ids,
            evidence
        )

        results.append({
            "claim": claim,
            "evidence_ids": evidence_ids,
            "status": verification["status"],
            "reason": verification["reason"]
        })

    return results