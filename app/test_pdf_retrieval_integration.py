
from pathlib import Path

import chromadb

from app import vector_store, retriever
from app.document_loader import load_pdf
from app.chunker import create_chunks


def test_pdf_to_chromadb_retrieval(tmp_path, monkeypatch):
    # Locate the existing PDF.
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "sWITCHING.pdf"

    assert pdf_path.is_file(), f"PDF not found: {pdf_path}"

    # Create an isolated temporary ChromaDB database.
    test_client = chromadb.PersistentClient(
        path=str(tmp_path / "test_chroma_db")
    )

    test_client.get_or_create_collection("documents")

    # Redirect both modules to the temporary database.
    monkeypatch.setattr(vector_store, "client", test_client)
    monkeypatch.setattr(retriever, "client", test_client)

    # Stage 1: Extract PDF pages.
    pages = load_pdf(str(pdf_path))

    assert pages
    assert all(page["text"].strip() for page in pages)

    # Stage 2: Split pages into chunks.
    chunks = create_chunks(pages)

    assert chunks
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all("page" in chunk for chunk in chunks)

    # Stage 3: Store chunks in ChromaDB.
    document_id = "integration_test_switching"

    vector_store.build_vector_database(
        chunks,
        document_id
    )

    collection = test_client.get_collection("documents")

    assert collection.count() == len(chunks)

    # Stage 4: Retrieve relevant chunks.
    results = retriever.retrieve_documents(
        query="Explain packet switching",
        document_id=document_id
    )

    assert results, "No relevant chunks were retrieved."

    assert any(
        "packet" in result["text"].lower()
        for result in results
    )

    # Verify that retrieved chunks retain their provenance.
    assert all(
        result["source"] == document_id
        for result in results
    )

    assert all(
        isinstance(result["page"], int) and result["page"] >= 1
        for result in results
    )
