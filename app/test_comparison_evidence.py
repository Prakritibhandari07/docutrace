
from app import cross_document_comparator


def test_compare_documents_returns_comparison_and_evidence(monkeypatch):
    expected_comparison = (
        "The OSI model has seven layers, while TCP/IP has four layers."
    )

    expected_evidence = [
        {
            "id": "E1",
            "document_id": "OSI and TCPIP.pdf",
            "document": "OSI and TCPIP.pdf",
            "page": 2,
            "text": "The OSI model consists of seven layers.",
        },
        {
            "id": "E2",
            "document_id": "OSI and TCPIP.pdf",
            "document": "OSI and TCPIP.pdf",
            "page": 3,
            "text": "The TCP/IP model consists of four layers.",
        },
    ]

    def mock_retrieve_comparison_evidence(
        query, document_ids, n_results
    ):
        return expected_evidence, {
            "OSI and TCPIP.pdf": 2,
            "sWITCHING.pdf": 0,
        }

    def mock_generate_comparison(query, context, evidence=None):
        return expected_comparison

    monkeypatch.setattr(
        cross_document_comparator,
        "retrieve_comparison_evidence",
        mock_retrieve_comparison_evidence,
    )
    monkeypatch.setattr(
        cross_document_comparator,
        "generate_comparison",
        mock_generate_comparison,
    )

    comparison, evidence = cross_document_comparator.compare_documents(
        "Compare the OSI model and TCP/IP model",
        ["OSI and TCPIP.pdf", "sWITCHING.pdf"],
    )

    assert expected_comparison in comparison
    assert "Evidence Limitations" in comparison
    assert "sWITCHING.pdf" in comparison
    assert evidence == expected_evidence
    assert len(evidence) == 2
    assert {item["id"] for item in evidence} == {"E1", "E2"}


def test_detect_contradictions_accepts_conflicting_claims_from_two_documents(
    monkeypatch,
):
    evidence = [
        {
            "id": "E1",
            "document_id": "Document A.pdf",
            "document": "Document A.pdf",
            "page": 2,
            "text": "The OSI model consists of seven layers.",
        },
        {
            "id": "E2",
            "document_id": "Document B.pdf",
            "document": "Document B.pdf",
            "page": 4,
            "text": "The OSI model consists of five layers.",
        },
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "The layer count differs: Document A reports "
                    "seven layers [E1], while Document B reports "
                    "five layers [E2]."
                )
            }

    monkeypatch.setattr(
        cross_document_comparator.requests,
        "post",
        lambda *args, **kwargs: MockResponse(),
    )

    result = cross_document_comparator.detect_contradictions(
        "Compare OSI layer counts",
        evidence,
    )

    assert "seven layers" in result
    assert "five layers" in result
    assert "[E1]" in result
    assert "[E2]" in result


def test_detect_contradictions_rejects_single_document_evidence():
    evidence = [
        {
            "id": "E1",
            "document_id": "Document A.pdf",
            "text": "The OSI model consists of seven layers.",
        },
        {
            "id": "E2",
            "document_id": "Document A.pdf",
            "text": "The OSI model has seven layers.",
        },
    ]

    result = cross_document_comparator.detect_contradictions(
        "Compare OSI layer counts",
        evidence,
    )

    assert result == ""


def test_detect_contradictions_rejects_invalid_citations(monkeypatch):
    evidence = [
        {
            "id": "E1",
            "document_id": "Document A.pdf",
            "document": "Document A.pdf",
            "page": 2,
            "text": "The OSI model consists of seven layers.",
        },
        {
            "id": "E2",
            "document_id": "Document B.pdf",
            "document": "Document B.pdf",
            "page": 4,
            "text": "The OSI model consists of five layers.",
        },
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": (
                    "The documents conflict [E1] [E99]."
                )
            }

    monkeypatch.setattr(
        cross_document_comparator.requests,
        "post",
        lambda *args, **kwargs: MockResponse(),
    )

    result = cross_document_comparator.detect_contradictions(
        "Compare OSI layer counts",
        evidence,
    )

    assert result == ""


def test_detect_contradictions_rejects_complementary_facts(monkeypatch):
    evidence = [
        {
            "id": "E1",
            "document_id": "Document A.pdf",
            "document": "Document A.pdf",
            "page": 2,
            "text": "The OSI model consists of seven layers.",
        },
        {
            "id": "E2",
            "document_id": "Document B.pdf",
            "document": "Document B.pdf",
            "page": 4,
            "text": "The TCP/IP model consists of four layers.",
        },
    ]

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "response": "No clear contradictions were identified."
            }

    monkeypatch.setattr(
        cross_document_comparator.requests,
        "post",
        lambda *args, **kwargs: MockResponse(),
    )

    result = cross_document_comparator.detect_contradictions(
        "Compare OSI and TCP/IP models",
        evidence,
    )

    assert result == "No clear contradictions were identified."

def test_generate_comparison_retries_when_one_sentence_is_uncited(
    monkeypatch,
):
    evidence = [
        {"id": "E1", "text": "The OSI model has seven layers."},
        {"id": "E2", "text": "TCP/IP has four layers."},
    ]

    responses = iter([
        "The OSI model has seven layers [E1]. TCP/IP has four layers.",
        "The OSI model has seven layers [E1]. TCP/IP has four layers [E2].",
    ])

    monkeypatch.setattr(
        cross_document_comparator.requests,
        "post",
        lambda *args, **kwargs: type(
            "Response",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"response": next(responses)},
            },
        )(),
    )

    result = cross_document_comparator.generate_comparison(
        "Compare OSI and TCP/IP",
        "Evidence context",
        evidence,
    )

    assert "[E1]" in result
    assert "[E2]" in result
    assert "TCP/IP has four layers [E2]" in result


def test_generate_comparison_rejects_uncited_sentence_after_retry(
    monkeypatch,
):
    responses = iter([
        "The OSI model has seven layers [E1]. Unsupported claim.",
        "The OSI model has seven layers [E1]. Still unsupported.",
    ])

    monkeypatch.setattr(
        cross_document_comparator.requests,
        "post",
        lambda *args, **kwargs: type(
            "Response",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {"response": next(responses)},
            },
        )(),
    )

    result = cross_document_comparator.generate_comparison(
        "Compare the models",
        "Evidence context",
        [{"id": "E1", "text": "The OSI model has seven layers."}],
    )

    assert "sentence-level citations" in result
