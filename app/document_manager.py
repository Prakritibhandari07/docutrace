import hashlib


def generate_document_id(file_bytes):
    """
    Generate a unique ID based on the contents of a PDF.
    """

    return hashlib.md5(
        file_bytes
    ).hexdigest()[:12]


def create_document_info(filename, file_bytes):
    """
    Create metadata for an uploaded document.
    """

    document_id = generate_document_id(
        file_bytes
    )

    return {
        "document_id": document_id,
        "filename": filename
    }