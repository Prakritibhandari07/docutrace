
from app.retriever import evidence_filter, select_best_evidence


def test_evidence_filter_removes_irrelevant_results():
    relevant = {
        "text": "Circuit switching establishes a dedicated path.",
        "relevance": 0.9,
        "keyword_overlap": 3,
        "page": 2,
    }
    irrelevant = {
        "text": "Completely unrelated material about cooking.",
        "relevance": 0.1,
        "keyword_overlap": 0,
        "page": 8,
    }

    filtered = evidence_filter([relevant, irrelevant])

    assert relevant in filtered
    assert irrelevant not in filtered


def test_select_best_evidence_ranks_results_by_relevance():
    lower_ranked = {
        "text": "Circuit switching reserves network resources.",
        "relevance": 0.8,
        "keyword_overlap": 2,
        "page": 3,
    }
    higher_ranked = {
        "text": "Circuit switching establishes a dedicated path.",
        "relevance": 0.9,
        "keyword_overlap": 3,
        "page": 2,
    }

    selected = select_best_evidence([lower_ranked, higher_ranked])

    assert selected[0] == higher_ranked
    assert lower_ranked in selected
