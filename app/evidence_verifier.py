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
    if not answer or not answer.strip():
        return []

    claims = []

    # Find sentences and their immediately following citations.
    sentence_pattern = re.compile(
        r"([^.!?]+[.!?])\s*((?:\[\s*E\d+\s*\]\s*)*)",
        re.IGNORECASE
    )

    matches = list(sentence_pattern.finditer(answer))

    for match in matches:
        claim_text = match.group(1).strip()
        citation_text = match.group(2)

        evidence_ids = re.findall(
            r"\[\s*(E\d+)\s*\]",
            citation_text,
            re.IGNORECASE
        )

        if claim_text:
            claims.append({
                "claim": claim_text,
                "evidence_ids": list(dict.fromkeys(
                    item.upper() for item in evidence_ids
                ))
            })

    # Handle citations placed at the end of the entire answer.
    # These cannot be reliably assigned to individual sentences.
    if not any(item["evidence_ids"] for item in claims):
        trailing_citations = re.search(
            r"((?:\[\s*E\d+\s*\]\s*)+)$",
            answer.strip(),
            re.IGNORECASE
        )

        if trailing_citations:
            evidence_ids = re.findall(
                r"\[\s*(E\d+)\s*\]",
                trailing_citations.group(1),
                re.IGNORECASE
            )

            evidence_ids = list(dict.fromkeys(
                item.upper() for item in evidence_ids
            ))

            text_without_citations = answer[
                :trailing_citations.start()
            ].strip()

            sentences = re.split(
                r"(?<=[.!?])\s+",
                text_without_citations
            )

            claims = [
                {
                    "claim": sentence.strip(),
                    "evidence_ids": evidence_ids.copy()
                }
                for sentence in sentences
                if sentence.strip()
            ]

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
    Calculate lexical overlap using normalized, unique words.
    This is a preliminary matching signal, not proof of entailment.
    """

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "in", "of", "to", "and", "for", "on", "that",
        "this", "with", "from", "it", "its", "be",
        "as", "by", "which", "where", "when", "does",
        "do", "has", "have", "had", "can", "will",
        "between", "before", "after", "during"
    }

    def get_words(text):
        words = re.findall(
            r"\b[a-zA-Z0-9]+\b",
            normalize_text(text)
        )

        return {
            word for word in words
            if word not in stop_words and len(word) > 2
        }

    claim_words = get_words(claim)
    evidence_words = get_words(evidence_text)

    if not claim_words:
        return None

    matched_words = claim_words.intersection(evidence_words)
    overlap_ratio = len(matched_words) / len(claim_words)

    print("[Verifier Debug] Claim:", claim)
    print("[Verifier Debug] Claim words:", sorted(claim_words))
    print("[Verifier Debug] Matched words:", sorted(matched_words))
    print("[Verifier Debug] Overlap:", round(overlap_ratio, 3))

    if overlap_ratio >= 0.60:
        return "Supported"

    if overlap_ratio >= 0.35:
        return "Partially Supported"

    return "Unsupported"

# ============================================================
# LLM VERIFICATION
# ============================================================


def llm_verify_claim(claim, evidence_items):
    print("NEW VERIFIER FUNCTION IS RUNNING")
    """
    Verify one claim against its explicitly cited evidence.
    Parse the model's status and reason robustly.
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

CLAIM:
{claim}

DOCUMENT EVIDENCE:
{context}

Rules:
1. Use only the supplied evidence.
2. A direct statement or faithful paraphrase counts as support.
3. Check important details, quantities, and conditions.
4. Supported means all important parts are supported.
5. Partially Supported means only some meaningful parts are supported.
6. Unsupported means the main assertion is not established.
7. Give a specific reason based on the evidence.

Return exactly:

STATUS: Supported

REASON: Explain which evidence supports or fails to support the claim.

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

        output = str(raw_response or "").strip()
        print("[Verification Debug] Raw model response:", repr(output))

        if not output:
            return {
                "status": "Verification Error",
                "reason": "The model returned an empty response.",
             }

    except Exception as exc:
        traceback.print_exc()
        return {
            "status": "Verification Error",
            "reason": f"Verification failed: {exc}",
        }

    # First, look for an explicit STATUS field.
    status_match = re.search(
        r"^\s*(?:\*\*)?\s*STATUS\s*:\s*"
        r"(Partially\s+Supported|Unsupported|Supported)\b",
        output,
        re.IGNORECASE | re.MULTILINE,
    )

    # Fallback: accept a response containing only a status.
    if not status_match:
        status_match = re.search(
            r"^\s*(Partially\s+Supported|Unsupported|Supported)\s*$",
            output,
            re.IGNORECASE | re.MULTILINE,
        )

    if not status_match:
        return {
            "status": "Verification Error",
            "reason": (
                "Could not identify a valid status in the model response: "
                + output[:250]
            ),
        }

    raw_status = re.sub(
        r"\s+",
        " ",
        status_match.group(1).strip()
    ).lower()

    if raw_status == "supported":
        status = "Supported"
    elif raw_status == "partially supported":
        status = "Partially Supported"
    else:
        status = "Unsupported"

    reason_match = re.search(
        r"^\s*(?:\*\*)?\s*REASON\s*:\s*(.*)",
        output,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )

    reason = (
        reason_match.group(1).strip().strip("*").strip()
        if reason_match and reason_match.group(1).strip()
        else "The model returned a status without a detailed reason."
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
            print("[Verifier Debug] Claim:", claim)
            print("[Verifier Debug] Evidence ID:", evidence_id)
            print("[Verifier Debug] Lexical status:", lexical_status)
            print("[Verifier Debug] Evidence text:", repr(evidence_text[:500]))
            print("CLAIM:", claim)
            print("EVIDENCE ID:", evidence_id)
            print("LEXICAL STATUS:", lexical_status)
            print("EVIDENCE TEXT:", evidence_text[:500])

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
                    print("[Verifier Debug] LLM fallback triggered for:", evidence_id)

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
