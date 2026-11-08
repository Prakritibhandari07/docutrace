from app import rag_pipeline


def test_run_rag_returns_insufficient_evidence(monkeypatch):
    llm_called = False

    def mock_retrieve(*args, **kwargs):
        return []

    def mock_generate_answer(*args, **kwargs):
        nonlocal llm_called
        llm_called = True
        return "Some answer"

    monkeypatch.setattr(
        rag_pipeline, "retrieve_documents", mock_retrieve
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer", mock_generate_answer
    )

    answer, evidence = rag_pipeline.run_rag(
        "What is artificial intelligence?",
        document_id="test-document",
    )

    assert "insufficient evidence" in answer.lower()
    assert "not find enough relevant information" in answer.lower()
    assert evidence == []
    assert llm_called is False



def test_run_rag_generates_answer_with_citation(monkeypatch):
    retrieved_documents = [
        {
            "source": "Agent.pdf",
            "page": 1,
            "text": "An agent perceives its environment through sensors.",
            "relevance": 0.85,
            "distance": 0.15,
        }
    ]

    captured_context = {}

    def mock_retrieve(*args, **kwargs):
        return retrieved_documents

    def mock_generate_answer(query, context):
        captured_context["query"] = query
        captured_context["context"] = context
        return "An agent perceives its environment through sensors."

    monkeypatch.setattr(
        rag_pipeline, "retrieve_documents", mock_retrieve
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer", mock_generate_answer
    )

    answer, evidence = rag_pipeline.run_rag(
        "How does an agent perceive its environment?",
        document_id="test-document",
    )

    assert "E1" in captured_context["context"]
    assert "Agent.pdf" in captured_context["context"]
    assert "[E1]" in answer
    assert len(evidence) == 1
    assert evidence[0]["id"] == "E1"

def test_rag_keeps_relevant_evidence_without_topic_filter(monkeypatch):
    retrieved_documents = [
        {
            "source": "Switching.pdf",
            "page": 1,
            "text": "Message switching stores the entire message before forwarding it.",
            "relevance": 0.80,
            "distance": 0.20,
        },
        {
            "source": "Switching.pdf",
            "page": 2,
            "text": "Packet switching divides data into smaller packets for transmission.",
            "relevance": 0.90,
            "distance": 0.10,
        },
    ]

    captured_context = {}

    def mock_retrieve(*args, **kwargs):
        return retrieved_documents

    def mock_generate_answer(query, context):
        captured_context["context"] = context
        return "Packet switching divides data into smaller packets."

    monkeypatch.setattr(
        rag_pipeline, "retrieve_documents", mock_retrieve
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer", mock_generate_answer
    )

    answer, evidence = rag_pipeline.run_rag(
        "Explain packet switching.",
        document_id="test-document",
    )

    context = captured_context["context"].lower()

    assert "packet switching" in context
    assert "message switching" in context
    assert len(evidence) == 2

def test_previous_page_context_is_cited(monkeypatch):
    retrieved_documents = [
        {
            "source": "Switching.pdf",
            "page": 8,
            "text": (
                "It allows simultaneous usage of the same "
                "channel by multiple users."
            ),
            "relevance": 0.90,
            "distance": 0.10,
        }
    ]

    class MockCollection:
        def get(self, **kwargs):
            return {
                "documents": [
                    "Packet Switched Network: Packet switching "
                    "divides data into smaller packets for transmission."
                ],
                "metadatas": [{"source": "Switching.pdf", "page": 7}],
            }

    monkeypatch.setattr(
        rag_pipeline, "get_collection",
        lambda: MockCollection()
    )
    monkeypatch.setattr(
        rag_pipeline, "retrieve_documents",
        lambda *args, **kwargs: retrieved_documents
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer",
        lambda query, context: (
            "Packet switching allows simultaneous usage "
            "of the same channel by multiple users."
        )
    )

    answer, evidence = rag_pipeline.run_rag(
        "What are the advantages of packet switching?",
        document_id="Switching.pdf",
    )

    assert len(evidence) == 2
    assert evidence[0]["page"] == 8
    assert evidence[1]["page"] == 7
    assert "[E1]" in answer
    assert "[E2]" in answer

def test_adjacent_context_does_not_duplicate_retrieved_evidence(monkeypatch):
    page_8_text = (
        "Advantages: It allows simultaneous usage of the same "
        "channel by multiple users."
    )

    retrieved_documents = [
        {
            "source": "Switching.pdf",
            "page": 8,
            "text": page_8_text,
            "relevance": 0.90,
            "distance": 0.10,
        },
        {
            "source": "Switching.pdf",
            "page": 9,
            "text": "It also supports different packet transmission techniques.",
            "relevance": 0.80,
            "distance": 0.20,
        },
    ]

    class MockCollection:
        def get(self, **kwargs):
            filters = kwargs.get("where", {}).get("$and", [])

            if any(item.get("page") == 8 for item in filters):
                return {
                    "documents": [page_8_text],
                    "metadatas": [
                        {"source": "Switching.pdf", "page": 8}
                    ],
                }

            return {
                "documents": [],
                "metadatas": [],
            }

    monkeypatch.setattr(
        rag_pipeline, "get_collection",
        lambda: MockCollection()
    )
    monkeypatch.setattr(
        rag_pipeline, "retrieve_documents",
        lambda *args, **kwargs: retrieved_documents
    )
    monkeypatch.setattr(
        rag_pipeline, "generate_answer",
        lambda query, context: "Packet switching has several advantages."
    )

    answer, evidence = rag_pipeline.run_rag(
        "What are the advantages of packet switching?",
        document_id="Switching.pdf",
    )

    page_8_evidence = [
        item for item in evidence
        if item["page"] == 8
    ]

    assert len(page_8_evidence) == 1
    assert len(evidence) == 2
    assert evidence[0]["context_for"] == ["E2"]
