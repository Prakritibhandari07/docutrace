
import hashlib
from app.document_manager import create_document_info


def test_create_document_info():
    filename = "test.pdf"
    file_bytes = b"this is a test PDF"

    document = create_document_info(filename, file_bytes)

    expected_id = hashlib.md5(file_bytes).hexdigest()[:12]

    assert document["filename"] == filename
    assert document["document_id"] == expected_id
    assert len(document["document_id"]) == 12


def test_same_content_produces_same_document_id():
    first = create_document_info("first.pdf", b"same content")
    second = create_document_info("second.pdf", b"same content")

    assert first["document_id"] == second["document_id"]
