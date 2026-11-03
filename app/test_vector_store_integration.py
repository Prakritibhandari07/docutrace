
import chromadb

from app import vector_store, retriever


def test_store_and_retrieve_document_chunks(tmp_path, monkeypatch):
    # Create an isolated temporary ChromaDB database
    test_client = chromadb.PersistentClient(
        path=str(tmp_path / "test_chroma_db")
    )

    test_client.get_or_create_collection("documents")

    # Redirect both modules to the temporary database
    monkeypatch.setattr(vector_store, "client", test_client)
    monkeypatch.setattr(retriever, "client", test_client)

    document_id = "test_switching_document"

    chunks = [
        {
            "text": (
                "Packet switching divides data into small packets. "
                "Each packet travels independently through the network "
                "and is reassembled at the destination."
            ),
            "page": 3,
        }
    ]

    # Store the chunks
    vector_store.build_vector_database(
        chunks,
        document_id
    )

    # Retrieve the stored chunks
    results = retriever.retrieve_documents(
        query="How does packet switching transfer data?",
        document_id=document_id
    )

    # Verify retrieval
    assert len(results) > 0

    assert any(
        result["source"] == document_id
        for result in results
    )

    assert any(
        result["page"] == 3
        for result in results
    )

    assert any(
        "packet switching" in result["text"].lower()
        for result in results
    )

    assert all(result["text"].strip() for result in results)
