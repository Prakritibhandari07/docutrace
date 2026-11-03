
from pathlib import Path

import pytest

from app.document_loader import load_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "data" / "sWITCHING.pdf"


def test_load_pdf_extracts_real_pdf():
    assert PDF_PATH.is_file(), f"Test PDF not found: {PDF_PATH}"

    documents = load_pdf(str(PDF_PATH))

    # Check that pages were actually extracted.
    assert len(documents) > 0

    # Check the structure of the extracted documents.
    assert all("page" in doc and "text" in doc for doc in documents)

    # Check that page numbers are valid and ordered.
    page_numbers = [doc["page"] for doc in documents]
    assert page_numbers == sorted(page_numbers)
    assert all(page >= 1 for page in page_numbers)

    # Check that extracted text is not empty.
    assert all(doc["text"].strip() for doc in documents)

    # Check that the expected subject appears in the PDF.
    full_text = " ".join(doc["text"] for doc in documents)
    assert "switching" in full_text.lower()


def test_load_pdf_raises_error_for_missing_file():
    with pytest.raises(FileNotFoundError):
        load_pdf("data/nonexistent_test_file.pdf")
