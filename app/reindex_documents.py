from app.document_loader import load_pdf
from app.chunker import create_chunks
from app.vector_store import delete_document, build_vector_database


DOCUMENTS = [
    {
        "name": "OSI and TCPIP.pdf",
        "path": "data/OSI and TCPIP.pdf",
    },
    {
        "name": "sWITCHING.pdf",
        "path": "data/sWITCHING.pdf",
    },
]


def reindex_document(document):

    filename = document["name"]
    filepath = document["path"]

    print(f"\nProcessing: {filename}")

    # Remove only the existing chunks
    # belonging to this document.
    delete_document(filename)

    # Extract PDF text.
    pages = load_pdf(filepath)

    # Create chunks.
    chunks = create_chunks(pages)

    # Store chunks using the filename
    # as the document identifier.
    build_vector_database(
        chunks,
        document_id=filename
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    print(
        f"Chunks indexed: {len(chunks)}"
    )

    print(
        f"Successfully re-indexed: {filename}"
    )


if __name__ == "__main__":

    for document in DOCUMENTS:

        reindex_document(
            document
        )

    print(
        "\nAll documents have been re-indexed."
    )