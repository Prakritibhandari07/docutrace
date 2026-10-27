from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for page in pages:

        page_chunks = splitter.split_text(
            page["text"]
        )
        for chunk in page_chunks:
            if "ensures better bandwidth usage" in chunk.lower():
                print("\n--- DEBUG CHUNK ---")
                print(chunk)
                print("--- END DEBUG CHUNK ---\n")

        for chunk in page_chunks:

            chunk = chunk.strip()

            if not chunk:
                continue

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks