
import json
from app import document_library


def test_add_document_and_load_again(tmp_path, monkeypatch):
    # Use a temporary JSON file instead of the real document library.
    documents_file = tmp_path / "documents.json"
    monkeypatch.setattr(
        document_library,
        "DOCUMENTS_FILE",
        str(documents_file),
    )

    test_document = {
        "document_id": "test123",
        "name": "test.pdf",
    }

    # Initially, no documents should exist.
    assert document_library.load_documents() == []

    # Add and verify the document.
    documents = document_library.add_document(test_document)
    assert documents == [test_document]

    # Reload from disk and verify persistence.
    with documents_file.open("r", encoding="utf-8") as file:
        saved_documents = json.load(file)

    assert saved_documents == [test_document]
    assert document_library.load_documents() == [test_document]

    # Adding the same name again should not create a duplicate.
    document_library.add_document(test_document)
    assert document_library.load_documents() == [test_document]
