from app.document_loader import load_pdf

pdf_path = "data/sWITCHING.pdf"

documents = load_pdf(pdf_path)

for doc in documents:
    print(f"\n--- Page {doc['page']} ---")
    print(doc["text"][:2000])