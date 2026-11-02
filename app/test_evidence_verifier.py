
from app import evidence_verifier


def test_supported_claims_are_verified(monkeypatch):
    answer = """
    Packet switching is a connectionless network switching technique. [E1]

    Each packet contains a header and a payload. [E1]

    The header contains addressing information. [E1]
    """

    evidence = [
        {
            "id": "E1",
            "document": "sWITCHING.pdf",
            "page": 7,
            "text": (
                "Packet switching is a connectionless network switching "
                "technique. The message is divided into packets. "
                "Each packet has two parts: a header and a payload. "
                "The header contains the addressing information."
            ),
        }
    ]

    # Avoid requiring a running Ollama server for the third claim.
    monkeypatch.setattr(
        evidence_verifier,
        "llm_verify_claim",
        lambda claim, evidence_items: {
            "status": "Supported",
            "reason": "Mocked LLM response for offline testing.",
        },
    )

    results = evidence_verifier.verify_answer(answer, evidence)

    assert len(results) == 3
    assert all(result["status"] == "Supported" for result in results)
    assert all("E1" in result["evidence_ids"] for result in results)

def test_verify_claim_preserves_llm_reason(monkeypatch):
    evidence = [
        {
            "id": "E1",
            "document": "sWITCHING.pdf",
            "page": 8,
            "text": (
                "It allows simultaneous usage of the same "
                "channel by multiple users."
            ),
        }
    ]

    expected_reason = (
        "The evidence establishes simultaneous channel sharing, "
        "but does not independently identify the subject."
    )

    monkeypatch.setattr(
        evidence_verifier,
        "llm_verify_claim",
        lambda claim, evidence_items: {
            "status": "Partially Supported",
            "reason": expected_reason,
        },
    )

    result = evidence_verifier.verify_claim(
        "Packet switching allows simultaneous usage of the same "
        "channel by multiple users.",
        ["E1"],
        evidence,
    )

    assert result["status"] == "Partially Supported"
    assert result["reason"] == expected_reason