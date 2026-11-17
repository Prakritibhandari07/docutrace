import re
import requests
import traceback

from app.llm_client import generate_text

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
        raw_response = generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=160,
            timeout=120,
        )
        output = raw_response.strip()
        print("[Verification Debug] Raw model response:", repr(output))

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "Verification Error",
            "reason": f"Verification failed: {exc}",
        }

    status_match = re.search(
        r"STATUS\s*:\s*(Supported|Partially Supported|Unsupported)\b",
        output,
        re.IGNORECASE,
    )

    
    # More flexible response parsing
    normalized_output = output.strip().lower()

    status_match = re.search(
        r"\b(partially\s+supported|unsupported|supported)\b",
        normalized_output,
    )

    if not status_match:
        return {
            "status": "Verification Error",
            "reason": f"Unrecognized model response: {output[:300]}",
        }

    raw_status = re.sub(
        r"\s+",
        " ",
        status_match.group(1)
    )

    if raw_status == "supported":
        status = "Supported"
    elif raw_status == "partially supported":
        status = "Partially Supported"
    else:
        status = "Unsupported"

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        output,
        re.IGNORECASE | re.DOTALL,
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match and reason_match.group(1).strip()
        else "The model did not provide a verification reason."
    )

    return {
        "status": status,
        "reason": reason,
    }


    if not status_match:
        return {
            "status": "Verification Error",
            "reason": "The model returned an invalid or unrecognized verification response.",
        }

    raw_status = status_match.group(1).lower()

    if raw_status == "supported":
        status = "Supported"
    elif raw_status == "partially supported":
        status = "Partially Supported"
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
    """Build a reason from the strongest matching evidence passage."""

    ignored_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "of", "to", "and", "or", "in", "on", "for",
        "that", "this", "it", "as", "by", "with",
        "its", "their", "which", "what", "can", "be"
    }

    claim_words = set(
        re.findall(r"\b[a-zA-Z0-9]+\b", normalize_text(claim))
    ) - ignored_words

    if not claim_words:
        return "The claim could not be meaningfully compared with the evidence."

    best_sentence = ""
    best_score = 0.0

    for item in evidence_items:
        text = str(item.get("text", ""))
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_words = set(
                re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    normalize_text(sentence)
                )
            ) - ignored_words

            if not sentence_words:
                continue

            score = len(claim_words & sentence_words) / len(claim_words)

            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()

    if best_sentence and best_score >= 0.5:
        return (
            f'The evidence states "{best_sentence}", '
            f'which contains relevant details from the claim.'
        )

    return (
        "The cited evidence does not provide a clear statement "
        "supporting the claim."
    )




def verify_claim(claim, evidence_ids, evidence):
    """
    Verify each citation independently.
    Use lexical matching first and the LLM only for ambiguous cases.
    """
    if not evidence_ids:
        return {
            "status": "Unsupported",
            "reason": "This claim has no cited evidence.",
            "evidence_ids": [],
            "partial_evidence_ids": [],
            "citation_results": [],
        }

    citation_results = []
    supported_ids = []
    partial_ids = []

    for evidence_id in evidence_ids:
        claim_evidence = get_claim_evidence([evidence_id], evidence)

        if not claim_evidence:
            citation_results.append({
                "evidence_id": evidence_id,
                "status": "Unsupported",
                "reason": "No matching evidence was found.",
            })
            continue

        # Include explicitly linked adjacent-page context.
        for item in evidence:
            if evidence_id in item.get("context_for", []):
                if item not in claim_evidence:
                    claim_evidence.append(item)

        evidence_text = "\n".join(
            str(item.get("text", ""))
            for item in claim_evidence
        )

        # First check exact wording.
        normalized_claim = normalize_text(claim)
        normalized_evidence = normalize_text(evidence_text)

        if normalized_claim and normalized_claim in normalized_evidence:
            status = "Supported"
            reason = "The evidence explicitly contains the claim's wording."

        else:
            # Check lexical overlap independently for this citation.
            lexical_status = direct_support_check(claim, evidence_text)

            if lexical_status == "Supported":
                status = "Supported"
                reason = (
                    "The cited passage contains strong overlap with "
                    "the important terms and details of the claim."
                )

            elif lexical_status == "Unsupported":
                status = "Unsupported"
                reason = (
                    "The cited passage does not contain enough of the "
                    "important terms and details of the claim."
                )

            else:
                # Use the LLM only when lexical matching is inconclusive.
                try:
                    llm_result = llm_verify_claim(
                        claim,
                        claim_evidence
                    )
                    status = llm_result.get(
                        "status",
                        "Verification Error"
                    )
                    reason = llm_result.get(
                        "reason",
                        "No verification reason was provided."
                    )

                    if status not in {
                        "Supported",
                        "Partially Supported",
                        "Unsupported",
                    }:
                        status = "Partially Supported"
                        reason = (
                            "The evidence has some matching details, "
                            "but semantic verification was inconclusive."
                        )

                except Exception:
                    status = "Partially Supported"
                    reason = (
                        "The evidence has some matching details, "
                        "but the language model could not complete verification."
                    )

        if status == "Supported":
            supported_ids.append(evidence_id)
        elif status == "Partially Supported":
            partial_ids.append(evidence_id)

        citation_results.append({
            "evidence_id": evidence_id,
            "status": status,
            "reason": reason,
        })

    if supported_ids:
        overall_status = "Supported"
        overall_reason = (
            "At least one cited passage independently supports this claim."
        )
    elif partial_ids:
        overall_status = "Partially Supported"
        overall_reason = (
            "The cited passages provide partial support for this claim."
        )
    else:
        overall_status = "Unsupported"
        overall_reason = (
            "None of the cited passages independently supports this claim."
        )

    return {
        "status": overall_status,
        "reason": overall_reason,
        "evidence_ids": supported_ids,
        "partial_evidence_ids": partial_ids,
        "citation_results": citation_results,
    }



def verify_answer(answer, evidence):
    """Verify each claim and preserve independently supported citations."""
    claims = extract_claims(answer)
    verification_results = []

    for claim_data in claims:
        claim = claim_data["claim"]
        evidence_ids = claim_data["evidence_ids"]

        result = verify_claim(claim, evidence_ids, evidence)

        result["claim"] = claim

        # Do not overwrite these with the original, unverified IDs.
        verification_results.append(result)

    return verification_results
