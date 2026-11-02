import chromadb
import re

client = chromadb.PersistentClient(path="chroma_db")


def get_collection():
    return client.get_collection("documents")


# --------------------------------------------------
# BASIC TEXT QUALITY CHECK
# --------------------------------------------------

def is_useful_text(text):
    if not text:
        return False

    text = text.strip()

    if len(text) < 50:
        return False

    alphanumeric_count = sum(char.isalnum() for char in text)
    return alphanumeric_count >= 30


# --------------------------------------------------
# QUERY KEYWORDS AND NORMALIZATION
# --------------------------------------------------

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an",
    "of", "in", "on", "to", "for", "its",
    "and", "or", "from", "these", "those",
    "this", "that", "does", "do",
    "how", "why", "which", "where", "when",
    "can", "could", "would",
    "about", "discussed", "provide",
    "information", "explain", "tell",
    "give", "me", "please"
}


def normalize_token(token):
    token = token.lower()

    # Categories -> category
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"

    # Switching -> switch
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]

        if len(token) >= 2 and token[-1] == token[-2]:
            token = token[:-1]

        return token

    # Processed -> process
    if len(token) > 4 and token.endswith("ed"):
        token = token[:-2]

        if len(token) >= 2 and token[-1] == token[-2]:
            token = token[:-1]

        return token

    # Classes -> class, switches -> switch
    if token.endswith(("sses", "ches", "shes", "xes", "zes")):
        return token[:-2]

    # Preserve words ending in ss
    if token.endswith("ss"):
        return token

    # Advantages -> advantage, models -> model
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]

    return token

def extract_query_keywords(query):
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())

    return {
        normalize_token(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    }


# --------------------------------------------------
# KEYWORD OVERLAP
# --------------------------------------------------

def keyword_overlap(query_keywords, text):
    text_words = {
        normalize_token(word)
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
    }

    if not query_keywords:
        return 0

    return len(query_keywords.intersection(text_words))

def query_phrase_overlap(query, text):
    query_words = [
        normalize_token(word)
        for word in re.findall(r"[a-zA-Z0-9]+", query.lower())
        if word not in STOP_WORDS and len(word) > 2
    ]

    text_words = [
        normalize_token(word)
        for word in re.findall(r"[a-zA-Z0-9]+", text.lower())
    ]

    if len(query_words) < 2:
        return 0

    query_phrases = {
        (query_words[i], query_words[i + 1])
        for i in range(len(query_words) - 1)
    }

    text_phrases = {
        (text_words[i], text_words[i + 1])
        for i in range(len(text_words) - 1)
    }

    return len(query_phrases.intersection(text_phrases))
# --------------------------------------------------
# SEMANTIC SCORE
# --------------------------------------------------

def semantic_score(distance):
    return 1 / (1 + max(distance, 0))


# --------------------------------------------------
# FINAL RELEVANCE SCORE
# --------------------------------------------------

def calculate_relevance(
    distance,
    overlap,
    total_keywords,
    phrase_overlap=0
):
    semantic = semantic_score(distance)

    keyword_score = (
        overlap / total_keywords
        if total_keywords > 0
        else 0
    )

    base_score = semantic * 0.70 + keyword_score * 0.30

    phrase_bonus = min(phrase_overlap * 0.025, 0.05)

    score = base_score + phrase_bonus

    return round(min(max(score, 0), 1), 3)

# --------------------------------------------------
# RETRIEVE FROM ONE DOCUMENT
# --------------------------------------------------

def retrieve_from_document(
    collection,
    query,
    document_id,
    query_keywords,
    n_results=6
):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"source": document_id}
        )
        print(f"[Retrieval Debug] Requested document: {document_id!r}")
        print(f"[Retrieval Debug] Returned metadata: {results.get('metadatas')}")
        print(f"[Retrieval Debug] Returned document count: {len(results.get('documents', [[]])[0])}")

    except Exception as e:
        print(f"Retrieval error for {document_id}: {e}")
        return []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for text, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if not is_useful_text(text):
            continue

        overlap = keyword_overlap(query_keywords, text)

        phrase_overlap = query_phrase_overlap(query, text)

        relevance = calculate_relevance(
            distance,
            overlap,
            len(query_keywords),
            phrase_overlap
        )

        retrieved.append({
            "text": text,
            "source": metadata.get("source", document_id),
            "page": metadata.get("page", "Unknown"),
            "distance": distance,
            "keyword_overlap": overlap,
            "phrase_overlap": phrase_overlap,
            "relevance": relevance
        })

    return retrieved


# --------------------------------------------------
# REMOVE DUPLICATE CHUNKS
# --------------------------------------------------


def remove_duplicates(documents):
    unique = []
    seen = set()

    for document in documents:
        text = document.get("text", "").strip()
        normalized = re.sub(r"\s+", " ", text.lower())

        # Preserve identical text from different sources or pages.
        key = (
            document.get("source", "Unknown"),
            document.get("page", "Unknown"),
            normalized
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(document)

    return unique


# --------------------------------------------------
# EVIDENCE FILTER
# --------------------------------------------------


def evidence_filter(documents):
    """Keep passages with meaningful relevance and keyword overlap."""
    if not documents:
        return []

    useful = []

    for document in documents:
        relevance = document.get("relevance", 0)
        overlap = document.get("keyword_overlap", 0)

        # Strong relevance still needs at least one matching keyword.
        if relevance >= 0.50 and overlap >= 2:
            useful.append(document)
        # Lower relevance needs stronger lexical support.
        elif relevance >= 0.45 and overlap >= 3:
            useful.append(document)

    return useful

# --------------------------------------------------
# SELECT BEST EVIDENCE
# --------------------------------------------------


def select_best_evidence(documents, max_documents=6):
    if not documents:
        return []

    filtered_documents = evidence_filter(documents)

    # Reject all passages if none meet the relevance criteria.
    if not filtered_documents:
        return []

    documents = remove_duplicates(filtered_documents)

    documents.sort(
        key=lambda item: (
            item.get("relevance", 0),
            item.get("keyword_overlap", 0)
        ),
        reverse=True
    )

    selected = []
    page_counts = {}

    for document in documents:
        source = document.get("source", "Unknown")
        page = document.get("page", "Unknown")
        page_key = (source, page)

        if page_counts.get(page_key, 0) >= 2:
            continue

        selected.append(document)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1

        if len(selected) >= max_documents:
            break

    return selected

# --------------------------------------------------
# MAIN RETRIEVER
# --------------------------------------------------

def retrieve_documents(
    query: str,
    document_id: str = None,
    document_ids: list = None,
    n_results: int = 12
):
    collection = get_collection()

    if document_ids:
        selected_documents = document_ids
    elif document_id:
        selected_documents = [document_id]
    else:
        selected_documents = None

    query_keywords = extract_query_keywords(query)
    all_documents = []

    # Search selected documents separately.
    if selected_documents:
        for selected_document in selected_documents:
            results = retrieve_from_document(
                collection=collection,
                query=query,
                document_id=selected_document,
                query_keywords=query_keywords,
                n_results=n_results
            )

            all_documents.extend(results)

    # Search the entire library.
    else:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for text, metadata, distance in zip(
            documents, metadatas, distances
        ):
            if not is_useful_text(text):
                continue

            overlap = keyword_overlap(query_keywords, text)
            phrase_overlap = query_phrase_overlap(query, text)

            relevance = calculate_relevance(
                distance,
                overlap,
                len(query_keywords),
                phrase_overlap
            )

            all_documents.append({
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "page": metadata.get("page", "Unknown"),
                "distance": distance,
                "keyword_overlap": overlap,
                "phrase_overlap": phrase_overlap,
                "relevance": relevance
            })

    if not all_documents:
        return []

    all_documents = remove_duplicates(all_documents)

    print("\n========== RETRIEVAL SCORES ==========")
    print("Query keywords:", query_keywords)

    for doc in sorted(
        all_documents,
        key=lambda item: item.get("relevance", 0),
        reverse=True
    ):
        print(
            f"\nPage: {doc.get('page')} | "
            f"Distance: {doc.get('distance'):.3f} | "
            f"Overlap: {doc.get('keyword_overlap')} | "
            f"Relevance: {doc.get('relevance')}"
        )
        print(doc.get("text", ""))

    print("======================================\n")

    return select_best_evidence(
        all_documents,
        max_documents=12
    )

    print("\n========== RETRIEVAL SCORES ==========")
    print("Query keywords:", query_keywords)

    for doc in sorted(
        all_documents,
        key=lambda item: item.get("relevance", 0),
        reverse=True
    ):
        print(
            f"\nPage: {doc.get('page')} | "
            f"Distance: {doc.get('distance'):.3f} | "
            f"Overlap: {doc.get('keyword_overlap')} | "
            f"Relevance: {doc.get('relevance')}"
        )
        print(doc.get("text", "")[:250])

    print("======================================\n")