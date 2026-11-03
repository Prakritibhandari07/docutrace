
import os
import re
from pathlib import Path

import chromadb
import pytest

from app import vector_store, retriever, rag_pipeline
from app.document_loader import load_pdf
from app.chunker import create_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "sWITCHING.pdf"


def prepare_test_database(tmp_path, monkeypatch, database_name):
    """Create an isolated ChromaDB database and process the test PDF."""

    assert PDF_PATH.is_file(), f"Test PDF not found: {PDF_PATH}"

    test_client = chromadb.PersistentClient(
        path=str(tmp_path / database_name)
    )

    test_client.get_or_create_collection("documents")

    monkeypatch.setattr(vector_store, "client", test_client)
    monkeypatch.setattr(retriever, "client", test_client)

    pages = load_pdf(str(PDF_PATH))
    assert pages, "No pages were extracted from the PDF."

    chunks = create_chunks(pages)
    assert chunks, "No chunks were created from the PDF."

    return test_client, chunks


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run real Ollama tests."
)
def test_real_pdf_to_ollama_rag(tmp_path, monkeypatch):

    # Prepare the PDF and isolated database.
    test_client, chunks = prepare_test_database(
        tmp_path,
        monkeypatch,
        "ollama_test_db"
    )

    document_id = "real_ollama_test"

    # Store the PDF chunks.
    vector_store.build_vector_database(
        chunks,
        document_id
    )

    collection = test_client.get_collection("documents")
    assert collection.count() > 0

    # Ask a question covered by the PDF.
    query = (
        "What advantages does packet switching provide "
        "for simultaneous channel sharing and bandwidth usage?"
    )

    # Run the real RAG pipeline and Ollama model.
    answer, evidence = rag_pipeline.run_rag(
        query,
        document_id=document_id
    )

    print("\n--- GENERATED ANSWER ---")
    print(answer)

    print("\n--- RETRIEVED EVIDENCE ---")

    for item in evidence:
        print(
            f"\nID: {item['id']} | "
            f"Page: {item['page']}"
        )
        print(item["text"])

    # Validate the answer and evidence.
    assert evidence, "No evidence was retrieved."
    assert answer and answer.strip(), "The model returned an empty answer."

    evidence_ids = {
        item["id"] for item in evidence
    }

    cited_ids = set(
        re.findall(r"\[(E\d+)\]", answer)
    )

    assert cited_ids, "The answer contains no evidence citations."

    assert cited_ids.issubset(evidence_ids), (
        "The answer contains citations not present in retrieved evidence."
    )


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS") != "1",
    reason="Set RUN_OLLAMA_TESTS=1 to run real Ollama tests."
)
def test_real_ollama_rejects_unsupported_question(tmp_path, monkeypatch):

    # Prepare the PDF and isolated database.
    _, chunks = prepare_test_database(
        tmp_path,
        monkeypatch,
        "unsupported_test_db"
    )

    document_id = "unsupported_question_test"

    # Store the PDF chunks.
    vector_store.build_vector_database(
        chunks,
        document_id
    )

    # Ask something that the PDF does not discuss.
    query = "Explain quantum computing and its applications."

    answer, evidence = rag_pipeline.run_rag(
        query,
        document_id=document_id
    )

    print("\n--- UNSUPPORTED QUESTION ---")
    print(query)

    print("\n--- GENERATED ANSWER ---")
    print(answer)

    print("\n--- RETRIEVED EVIDENCE ---")

    for item in evidence:
        print(f"Page: {item['page']}")
        print(item["text"])

    # The pipeline must return a response.
    assert answer and answer.strip(), (
        "The model returned an empty answer."
    )

    # Verify that DocuTrace rejects unsupported questions.
    assert "insufficient evidence" in answer.lower(), (
        "DocuTrace did not identify the lack of sufficient evidence."
    )

    assert (
        "could not find enough relevant information" in answer.lower()
        or "not enough evidence" in answer.lower()
    ), (
        "DocuTrace did not explain why it could not answer."
    )