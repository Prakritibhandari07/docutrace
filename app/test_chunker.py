
from app.chunker import create_chunks


def test_create_chunks_returns_text_and_page():
    pages = [
        {
            "page": 3,
            "text": "Circuit switching establishes a dedicated path."
        }
    ]

    chunks = create_chunks(pages)

    assert len(chunks) == 1
    assert chunks[0]["page"] == 3
    assert "Circuit switching" in chunks[0]["text"]


def test_create_chunks_splits_long_text():
    pages = [
        {
            "page": 1,
            "text": "Packet switching transfers data. " * 100
        }
    ]

    chunks = create_chunks(pages)

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 900 for chunk in chunks)


def test_create_chunks_preserves_page_numbers():
    pages = [
        {"page": 1, "text": "Circuit switching information. " * 20},
        {"page": 2, "text": "Packet switching information. " * 20},
    ]

    chunks = create_chunks(pages)

    page_numbers = {chunk["page"] for chunk in chunks}

    assert page_numbers == {1, 2}


def test_create_chunks_ignores_empty_pages():
    pages = [
        {"page": 1, "text": ""},
        {"page": 2, "text": "   "},
    ]

    chunks = create_chunks(pages)

    assert chunks == []


def test_create_chunks_handles_empty_input():
    assert create_chunks([]) == []
