from app.document_analysis import (
    clean_words,
    extract_keywords,
    find_keyword_pages,
    detect_headings,
    sentence_score,
    extractive_summary,
)


def test_clean_words_removes_stop_words_and_short_words():
    result = clean_words("The quick fox is in a forest")

    assert result == ["quick", "fox", "forest"]


def test_extract_keywords_returns_most_frequent_words():
    result = extract_keywords(
        "python python testing testing testing documents",
        limit=2
    )

    assert result == ["testing", "python"]


def test_find_keyword_pages_finds_matching_pages():
    pages = [
        {"page": 1, "text": "Python is useful"},
        {"page": 2, "text": "Document analysis with Python"},
        {"page": 3, "text": "Java programming"},
    ]

    result = find_keyword_pages(pages, ["python", "java"])

    assert result == {
        "python": [1, 2],
        "java": [3],
    }


def test_detect_headings_extracts_heading_lines():
    pages = [
        {
            "page": 1,
            "text": "INTRODUCTION\nthis is a longer paragraph with details.\nConclusion:"
        }
    ]

    result = detect_headings(pages)

    assert result == [
        {"text": "INTRODUCTION", "page": 1},
        {"text": "Conclusion:", "page": 1},
    ]


def test_sentence_score_counts_unique_keyword_matches():
    result = sentence_score(
        "Python supports Python testing and document analysis.",
        ["python", "testing", "analysis"]
    )

    assert result == 3


def test_extractive_summary_selects_relevant_sentences_in_original_order():
    text = (
        "Python is a programming language used for many different applications. "
        "The weather is pleasant today and the sky is very clear. "
        "Python supports document analysis and automated testing."
    )

    result = extractive_summary(
        text,
        ["python", "document", "analysis", "testing"],
        max_sentences=2
    )

    assert result == [
        "Python is a programming language used for many different applications.",
        "Python supports document analysis and automated testing.",
    ]
