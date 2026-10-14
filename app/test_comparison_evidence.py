from app.cross_document_comparator import compare_documents

comparison, evidence = compare_documents(
    "Compare the OSI model and TCP/IP model",
    ["OSI and TCPIP.pdf", "sWITCHING.pdf"]
)

print("\n========== COMPARISON RESULT ==========\n")
print(comparison)

print("\n========== RETRIEVED EVIDENCE ==========\n")

for item in evidence:
    print(f"Evidence ID: {item['id']}")
    print(f"Document: {item['document']}")
    print(f"Page: {item['page']}")
    print(f"Passage:\n{item['text']}")
    print("-" * 60)

print("\nTotal evidence passages:", len(evidence))