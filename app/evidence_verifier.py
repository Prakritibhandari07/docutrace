import re
import requests
import traceback


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

Verify exactly ONE claim against the supplied evidence.

CLAIM TO VERIFY:
{claim}

DOCUMENT EVIDENCE:
{context}

RULES:
1. Use only the supplied evidence.
2. Determine whether the evidence supports the meaning of this exact claim.
3. A direct statement or faithful paraphrase counts as support.
4. Check negation, quantities, conditions, and other important details.
5. Do not evaluate other claims or facts.
6. If all important parts are supported, return Supported.
7. If only some meaningful parts are supported, return Partially Supported.
8. If the main assertion is not established, return Unsupported.
9. Write a reason specific to THIS claim.
10. The reason must identify the relevant statement in the evidence
    and explain how it supports or fails to support the claim.
11. Never use evidence about a different claim to justify this claim.
12. Do not repeat a generic reason for every claim.

Return exactly this format:

STATUS: Supported
REASON: The evidence states [specific relevant statement], which supports the claim because [brief explanation].

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
                    "num_predict": 160,
                },
            },
            timeout=120,
        )

        response.raise_for_status()
        output = response.json().get("response", "").strip()

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "Unsupported",
            "reason": f"Verification failed: {exc}",
        }

    status_match = re.search(
        r"STATUS\s*:\s*(Supported|Partially Supported|Unsupported)\b",
        output,
        re.IGNORECASE,
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        output,
        re.IGNORECASE | re.DOTALL,
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
        if reason_match and reason_match.group(1).strip()
        else "The model did not provide a valid verification reason."
    )

    return {
        "status": status,
        "reason": reason,
    }


def build_evidence_reason(claim, evidence_items):
    """Build a reason using the actual cited evidence."""

    claim_words = set(normalize_text(claim).split())
    best_sentence = ""
    best_score = 0

    for item in evidence_items:
        text = str(item.get("text", ""))

        # Split the passage into sentences.
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            normalized_sentence = normalize_text(sentence)
            sentence_words = set(normalized_sentence.split())

            if not sentence_words:
                continue

            # Ignore very common words when comparing meanings.
            ignored_words = {
                "the", "a", "an", "is", "are", "was", "were",
                "of", "to", "and", "or", "in", "on", "for",
                "that", "this", "it", "as", "by", "with"
            }

            meaningful_claim_words = claim_words - ignored_words
            meaningful_sentence_words = sentence_words - ignored_words

            if not meaningful_claim_words:
                continue

            score = len(
                meaningful_claim_words & meaningful_sentence_words
            ) / len(meaningful_claim_words)

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

    if best_sentence and best_score >= 0.5:
        return (
            f'The cited evidence states "{best_sentence}", '
            f'which shares key details with the claim.'
        )

    return "The cited passage does not clearly match the claim's wording."



def verify_claim(claim, evidence_ids, evidence):

    print("NEW VERIFY_CLAIM FUNCTION IS RUNNING")
    """Verify a claim against its cited evidence."""

    if not evidence_ids:
        return {
            "status": "Unsupported",
            "reason": "This claim has no evidence citation."
        }

    claim_evidence = get_claim_evidence(
        evidence_ids, evidence
    )

    if not claim_evidence:
        return {
            "status": "Unsupported",
            "reason": "No matching evidence was found for the cited IDs."
        }

    evidence_text = "\n".join(
        str(item.get("text", ""))
        for item in claim_evidence
    )

    normalized_claim = normalize_text(claim)
    normalized_evidence = normalize_text(evidence_text)

    # Direct wording match: the claim is explicitly in the evidence.
    if normalized_claim and normalized_claim in normalized_evidence:
        return {
            "status": "Supported",
            "reason": "The cited evidence explicitly contains the claim's wording."
        }

    # Otherwise, use the LLM verifier.
    result = llm_verify_claim(claim, claim_evidence)

    status = str(result.get("status", "")).strip().lower()

    if status == "supported":
        result["status"] = "Supported"
    elif status in {"partially supported", "partially_supported"}:
        result["status"] = "Partially Supported"
    else:
        result["status"] = "Unsupported"
 
    if not result.get("reason"):
        result["reason"] = build_evidence_reason(
            claim, claim_evidence
        )

    return result


def verify_answer(answer, evidence):
    """Verify each claim in an answer against its cited evidence."""

    claims = extract_claims(answer)
    verification_results = []

    for claim_data in claims:
        claim = claim_data["claim"]
        evidence_ids = claim_data["evidence_ids"]

        result = verify_claim(
            claim,
            evidence_ids,
            evidence
        )

        result["claim"] = claim
        result["evidence_ids"] = evidence_ids

        verification_results.append(result)

    return verification_results
