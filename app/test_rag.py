import hashlib
from pathlib import Path

from app.rag_pipeline import run_rag


# Automatically compute the same document_id the Streamlit app would
# generate for this file, based on its contents.
pdf_path = Path("data") / "Agent.pdf"

file_bytes = pdf_path.read_bytes()

document_id = hashlib.md5(file_bytes).hexdigest()[:12]

print(f"Using document_id: {document_id}")


question = "What are the types of agents?"


result = run_rag(
    question,
    document_id,
    top_k=2
)


print("\nANSWER:")
print(result["answer"])


print("\nSOURCES:")

for source in result["sources"]:

    print(f"\nPage {source['page']}")
    print(source["text"][:300])