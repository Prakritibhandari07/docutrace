from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
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

            chunk = chunk.strip()

            if not chunk:
                continue

            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks