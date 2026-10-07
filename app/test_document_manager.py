from app.document_manager import create_document_info


file_bytes = b"this is a test PDF"


document = create_document_info(
    "test.pdf",
    file_bytes
)


print("\nDOCUMENT INFO:")
print(document)