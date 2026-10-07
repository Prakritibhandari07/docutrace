import json
import os

DOCUMENTS_FILE = "data/documents.json"


def load_documents():
    """Load documents from JSON file."""
    if not os.path.exists(DOCUMENTS_FILE):
        return []

    try:
        with open(DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except (json.JSONDecodeError, FileNotFoundError):
        return []


def save_documents(documents):
    """Save documents list to JSON file safely."""
    os.makedirs("data", exist_ok=True)

    with open(DOCUMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2)


def add_document(document):
    """Add document only if it does not already exist."""

    documents = load_documents()

    document_name = document.get("name")

    # Check whether document already exists
    for existing_document in documents:
        if existing_document.get("name") == document_name:
            return documents

    documents.append(document)

    save_documents(documents)

    return documents