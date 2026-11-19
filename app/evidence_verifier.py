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

    # Process each paragraph independently.
    paragraphs = re.split(r"\n\s*\n", answer.strip())

    sentence_pattern = re.compile(
        r"([^.!?]+[.!?])\s*((?:\[\s*E\d+\s*\]\s*)*)",
        re.IGNORECASE
    )

    for paragraph in paragraphs:
        matches = list(sentence_pattern.finditer(paragraph))

        if not matches:
            continue

        paragraph_ids = list(dict.fromkeys(
            item.upper()
            for item in re.findall(
                r"\[\s*(E\d+)\s*\]",
                paragraph,
                re.IGNORECASE
            )
        ))

        # Inherit a single unique paragraph citation as a
        # candidate for uncited sentences. Each claim is
        # still independently checked by the verifier.
        inherited_ids = paragraph_ids if len(paragraph_ids) == 1 else []

        for match in matches:
            claim_text = match.group(1).strip()

            evidence_ids = list(dict.fromkeys(
                item.upper()
                for item in re.findall(
                    r"\[\s*(E\d+)\s*\]",
                    match.group(2),
                    re.IGNORECASE
                )
            ))

            if not evidence_ids:
                evidence_ids = inherited_ids.copy()

            if claim_text:
                claims.append({
                    "claim": claim_text,
                    "evidence_ids": evidence_ids
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
    """Verify one claim against its explicitly cited evidence."""

    evidence_parts = []

    for item in evidence_items:
        evidence_parts.append(
            f"[{item.get('id', 'UNKNOWN')}]\n"
            f"{item.get('text', '')}"
        )

    context = "\n\n".join(evidence_parts)

    prompt = f"""
You are a strict document evidence verifier.

CLAIM:
{claim}

EVIDENCE:
{context}

Decide whether the evidence supports the complete claim.

Return exactly ONE of these three labels, with no explanation:
Supported
Partially Supported
Unsupported

Supported = all important parts are supported.
Partially Supported = some important parts are supported.
Unsupported = the main assertion is not established.
"""

    try:
        output = generate_text(
            prompt=prompt,
            temperature=0.0,
            max_tokens=30,
            timeout=120,
        )

        output = str(output or "").strip()
        print("[Verification Debug] Raw model response:", repr(output))

        match = re.search(
            r"\b(Partially\s+Supported|Unsupported|Supported)\b",
            output,
            re.IGNORECASE,
        )

        if not match:
            return {
                "status": "Verification Error",
                "reason": "Unrecognized verifier response: " + repr(output[:250]),
            }

        normalized = re.sub(r"\s+", " ", match.group(1)).lower()

        if normalized == "supported":
            status = "Supported"
        elif normalized == "partially supported":
            status = "Partially Supported"
        else:
            status = "Unsupported"

        return {
            "status": status,
            "reason": "The evidence was evaluated against the claim.",
        }

    except Exception as exc:
        return {
            "status": "Verification Error",
            "reason": f"Verification failed: {exc}",
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
    Verify each citation independently using the LLM.
    Lexical overlap is not sufficient to establish support.
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
    error_ids = []

    for evidence_id in evidence_ids:
        claim_evidence = get_claim_evidence([evidence_id], evidence)

        if not claim_evidence:
            citation_results.append({
                "evidence_id": evidence_id,
                "status": "Unsupported",
                "reason": "No matching evidence was found.",
            })
            continue

        for item in evidence:
            if evidence_id.upper() in [
                str(value).upper()
                for value in item.get("context_for", [])
            ]:
                if item not in claim_evidence:
                    claim_evidence.append(item)

        try:
            llm_result = llm_verify_claim(claim, claim_evidence)

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
                "Verification Error",
            }:
                status = "Verification Error"

        except Exception as exc:
            status = "Verification Error"
            reason = f"Verification failed: {exc}"

        if status == "Supported":
            supported_ids.append(evidence_id)
        elif status == "Partially Supported":
            partial_ids.append(evidence_id)
        elif status == "Verification Error":
            error_ids.append(evidence_id)

        citation_results.append({
            "evidence_id": evidence_id,
            "status": status,
            "reason": reason,
        })

    if supported_ids:
        overall_status = "Supported"
        overall_reason = next(
            item["reason"]
            for item in citation_results
            if item["status"] == "Supported"
        )
    elif partial_ids:
        overall_status = "Partially Supported"
        overall_reason = next(
            item["reason"]
            for item in citation_results
            if item["status"] == "Partially Supported"
        )
    elif error_ids:
        overall_status = "Verification Error"
        overall_reason = next(
            item["reason"]
            for item in citation_results
            if item["status"] == "Verification Error"
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
