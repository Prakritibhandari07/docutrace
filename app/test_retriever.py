
from app import retriever


# -----------------------------------------
# TEXT QUALITY TESTS
# -----------------------------------------

def test_is_useful_text_accepts_valid_text():
    text = (
        "Packet switching transfers data through "
        "independent packets across a network."
    )
    assert retriever.is_useful_text(text) is True


def test_is_useful_text_rejects_empty_text():
    assert retriever.is_useful_text("") is False


def test_is_useful_text_rejects_short_text():
    assert retriever.is_useful_text("Short text") is False


# -----------------------------------------
# KEYWORD EXTRACTION TESTS
# -----------------------------------------

def test_extract_query_keywords():
    keywords = retriever.extract_query_keywords(
        "What is the accuracy of the MNIST model?"
    )

    assert keywords == {"accuracy", "mnist", "model"}


def test_normalize_token():
    assert retriever.normalize_token("models") == "model"
    assert retriever.normalize_token("switching") == "switch"
    assert retriever.normalize_token("categories") == "category"


# -----------------------------------------
# KEYWORD OVERLAP TESTS
# -----------------------------------------

def test_keyword_overlap_counts_matching_words():
    keywords = {"packet", "switch", "network"}

    text = "Packet switching transfers data through a network."

    assert retriever.keyword_overlap(keywords, text) == 3


def test_keyword_overlap_returns_zero_without_matches():
    keywords = {"database", "computer"}

    text = "Packet switching transfers information."

    assert retriever.keyword_overlap(keywords, text) == 0


# -----------------------------------------
# RELEVANCE SCORING TESTS
# -----------------------------------------

def test_semantic_score():
    score = retriever.semantic_score(0.15)

    assert round(score, 3) == 0.870


def test_semantic_score_does_not_exceed_one_for_negative_distance():
    assert retriever.semantic_score(-1) == 1


def test_calculate_relevance():
    score = retriever.calculate_relevance(
        distance=0.15,
        overlap=2,
        total_keywords=2
    )

    assert score == 0.909


# -----------------------------------------
# EVIDENCE FILTER TESTS
# -----------------------------------------

def test_evidence_filter_accepts_strong_evidence():
    documents = [
        {
            "text": "Relevant passage",
            "relevance": 0.60,
            "keyword_overlap": 2
        }
    ]

    results = retriever.evidence_filter(documents)

    assert len(results) == 1


def test_evidence_filter_rejects_weak_evidence():
    documents = [
        {
            "text": "Weak passage",
            "relevance": 0.40,
            "keyword_overlap": 1
        }
    ]

    assert retriever.evidence_filter(documents) == []


def test_evidence_filter_accepts_lower_score_with_more_overlap():
    documents = [
        {
            "text": "Passage with strong keyword overlap",
            "relevance": 0.45,
            "keyword_overlap": 3
        }
    ]

    assert len(retriever.evidence_filter(documents)) == 1


# -----------------------------------------
# DUPLICATE REMOVAL TESTS
# -----------------------------------------

def test_remove_duplicates():
    documents = [
        {"text": "Packet switching transfers data."},
        {"text": "packet switching transfers data."},
        {"text": "Circuit switching establishes a path."}
    ]

    results = retriever.remove_duplicates(documents)

    assert len(results) == 2


# -----------------------------------------
# BEST EVIDENCE SELECTION TESTS
# -----------------------------------------

def test_select_best_evidence_limits_chunks_per_page():
    documents = [
        {
            "text": f"Evidence passage number {i}",
            "source": "doc1",
            "page": 1,
            "relevance": 0.90 - (i * 0.05),
            "keyword_overlap": 3
        }
        for i in range(3)
    ]

    documents.append({
        "text": "Evidence from another page",
        "source": "doc1",
        "page": 2,
        "relevance": 0.70,
        "keyword_overlap": 3
    })

    results = retriever.select_best_evidence(
        documents,
        max_documents=6
    )

    page_one_count = sum(
        doc["page"] == 1 for doc in results
    )

    assert page_one_count <= 2


def test_select_best_evidence_respects_max_documents():
    documents = [
        {
            "text": f"Unique evidence passage number {i}",
            "source": "doc1",
            "page": i,
            "relevance": 0.80,
            "keyword_overlap": 3
        }
        for i in range(5)
    ]

    results = retriever.select_best_evidence(
        documents,
        max_documents=3
    )

    assert len(results) == 3


def test_select_best_evidence_returns_empty_when_no_evidence_passes():
    documents = [
        {
            "text": "Unrelated passage about cooking food.",
            "source": "doc1",
            "page": 1,
            "relevance": 0.30,
            "keyword_overlap": 0
        },
        {
            "text": "Unrelated passage about sports.",
            "source": "doc1",
            "page": 2,
            "relevance": 0.20,
            "keyword_overlap": 0
        }
    ]

    results = retriever.select_best_evidence(documents)

    assert results == []


def test_remove_duplicates_across_different_documents():
    documents = [
        {
            "text": "Packet switching transfers data.",
            "source": "doc1",
            "page": 1
        },
        {
            "text": "Packet switching transfers data.",
            "source": "doc2",
            "page": 4
        }
    ]

    results = retriever.remove_duplicates(documents)

    # Check whether provenance from both documents is preserved.
    assert len(results) == 2


def test_normalize_token_does_not_corrupt_unrelated_words():
    assert retriever.normalize_token("classes") == "class"
    assert retriever.normalize_token("business") == "business"

# -----------------------------------------
# RETRIEVAL QUALITY TESTS
# -----------------------------------------

def test_packet_switching_ranks_above_message_switching():
    query = "Explain advantages of packet switching"

    keywords = retriever.extract_query_keywords(query)

    packet_passage = (
        "Packet switching allows simultaneous usage of the "
        "same channel by multiple users. It ensures better "
        "bandwidth usage because packets from multiple sources "
        "can be transferred through the same link."
    )

    message_passage = (
        "In this method unlike circuit switched network, "
        "resources are shared among multiple users. "
        "Messages are stored and forwarded between nodes."
    )

    packet_overlap = retriever.keyword_overlap(
        keywords, packet_passage
    )

    message_overlap = retriever.keyword_overlap(
        keywords, message_passage
    )

    packet_score = retriever.calculate_relevance(
        distance=0.798,
        overlap=packet_overlap,
        total_keywords=len(keywords)
    )

    message_score = retriever.calculate_relevance(
        distance=0.727,
        overlap=message_overlap,
        total_keywords=len(keywords)
    )

    assert packet_score > message_score


def test_unrelated_query_has_no_keyword_overlap():
    query = "Explain quantum computing and its applications"

    keywords = retriever.extract_query_keywords(query)

    passage = (
        "Packet switching transfers data through independent "
        "packets across a communication network."
    )

    assert retriever.keyword_overlap(keywords, passage) == 0


def test_specific_topic_terms_are_recognized():
    keywords = retriever.extract_query_keywords(
        "Explain advantages of packet switching"
    )

    assert "packet" in keywords
    assert "switch" in keywords
    assert "advantage" in keywords



def test_query_phrase_matching():
    query = "Explain advantages of packet switching"

    packet_text = (
        "Packet switching divides messages into "
        "individual packets."
    )

    message_text = (
        "Message switching stores and forwards "
        "messages between network nodes."
    )

    assert retriever.query_phrase_overlap(
        query, packet_text
    ) > retriever.query_phrase_overlap(
        query, message_text
    )