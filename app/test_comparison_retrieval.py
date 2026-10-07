from app.retriever import (
    retrieve_from_document,
    get_collection,
    extract_query_keywords,
    evidence_filter,
    select_best_evidence,
)

query = "Explain circuit switching and its characteristics."
document_name = "sWITCHING.pdf"

collection = get_collection()
query_keywords = extract_query_keywords(query)

raw_results = retrieve_from_document(
    collection=collection,
    query=query,
    document_id=document_name,
    query_keywords=query_keywords,
    n_results=10,
)

print("\nRaw results:", len(raw_results))

filtered_results = evidence_filter(raw_results)
print("After evidence filter:", len(filtered_results))

selected_results = select_best_evidence(raw_results)
print("After evidence selection:", len(selected_results))

for result in selected_results:
    print(
        f"\nPage: {result['page']}"
        f"\nScore: {result['relevance']}"
        f"\nOverlap: {result['keyword_overlap']}"
        f"\nText: {result['text'][:800]}"
    )