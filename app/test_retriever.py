from app.retriever import retrieve_documents


question = "What is the accuracy of the MNIST model?"

results = retrieve_documents(
    question,
    top_k=3
)


print("\nRETRIEVED DOCUMENTS:\n")


for i, result in enumerate(results, start=1):

    print(f"--- RESULT {i} ---")
    print(f"Page: {result['page']}")
    print(f"Distance: {result['distance']:.4f}")
    print(result["text"])
    print()