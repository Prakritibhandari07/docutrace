from app.document_library import (
    load_documents,
    add_document
)


test_document = {
    "document_id": "test123",
    "filename": "test.pdf"
}


print("\nBefore adding:")
print(load_documents())


documents = add_document(
    test_document
)


print("\nAfter adding:")
print(documents)


print("\nLoaded again:")
print(load_documents())