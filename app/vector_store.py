import chromadb


client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_or_create_collection(
    "documents"
)


def reset_vector_database():
    """
    Delete and recreate the entire vector database.

    Use only when a complete database reset is intentionally required.
    """
    client.delete_collection("documents")

    global collection

    collection = client.get_or_create_collection(
        "documents"
    )

    return collection


def delete_document(document_id: str):
    """
    Delete all chunks belonging to one document.
    """
    global collection

    try:
        collection = client.get_collection(
            "documents"
        )
    except Exception:
        collection = client.get_or_create_collection(
            "documents"
        )

    collection.delete(
        where={
            "source": document_id
        }
    )


def build_vector_database(
    chunks,
    document_id: str
):
    """
    Add document chunks to the Chroma vector database.

    Each chunk stores:
    - source: document identifier
    - page: PDF page number
    """

    global collection

    try:
        collection = client.get_collection(
            "documents"
        )

    except Exception:
        collection = client.get_or_create_collection(
            "documents"
        )

    for i, chunk in enumerate(chunks):

        if isinstance(chunk, dict):

            text = chunk.get(
                "text",
                ""
            )

            page = chunk.get(
                "page",
                None
            )

        else:

            text = str(chunk)
            page = None

        text = text.strip()

        if not text:
            continue

        collection.add(
            documents=[text],

            metadatas=[
                {
                    "source": document_id,
                    "page": page
                }
            ],

            ids=[
                f"{document_id}_{i}"
            ]
        )