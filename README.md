# DocuTrace

### Search. Compare. Trace the Evidence.

DocuTrace is a Python and Streamlit application for searching PDF documents, answering questions using document evidence, and comparing information across multiple documents.

It uses Retrieval-Augmented Generation (RAG), ChromaDB for vector search, and a locally running Ollama language model to generate evidence-backed answers with page-level citations.

## Features

- Upload and process PDF documents
- Search document content using semantic retrieval
- Ask questions about selected documents
- Generate answers using RAG and a local language model
- Display supporting evidence and page-level citations
- Compare information across multiple documents
- Verify answer claims against retrieved evidence
- Manage and index a document library

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Ollama
- Sentence Transformers
- PyPDF
- pytest

## Architecture

```text
PDF Documents
      |
      v
Document Loader
      |
      v
Text Cleaning and Chunking
      |
      v
Embeddings and ChromaDB
      |
      v
Semantic Retrieval
      |
      v
RAG Pipeline
      |
      v
Ollama LLM
      |
      v
Answer + Evidence + Citations
```

## Prerequisites

- Python 3.11
- [Ollama](https://ollama.com/) installed and running
- Git

## Installation

Clone the repository:

```bash
git clone https://github.com/Prakritibhandari07/docutrace.git
cd docutrace
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the language model:

```powershell
ollama pull qwen2.5:3b
```

## Run the Application

From the project root, run:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

Open the local URL displayed in the terminal.

## Usage

1. Start the application.
2. Upload a PDF or select a document from the library.
3. Ask a question about the selected document.
4. Review the answer, supporting evidence, and citations.
5. Use the comparison section to compare information across multiple documents.

## Application Screenshots

### 1. Document Library

![DocuTrace Document Library](screenshots/document-library.png)

### 2. Question Answering

![DocuTrace Question Answering](screenshots/question-answering.png)

### 3. Supporting Evidence and Citations

![DocuTrace Supporting Evidence](screenshots/supporting-evidence.png)

## Local Processing

DocuTrace uses a local Ollama model for answer generation. Actual processing and data handling depend on the configured components and application settings.

## Testing

Run the available tests from the project root:

```powershell
python -m pytest
```

## Author

**Prakriti Bhandari**

- GitHub: [Prakritibhandari07](https://github.com/Prakritibhandari07)
- LinkedIn: [Prakriti Bhandari](https://www.linkedin.com/in/prakriti-bhandari-18pp)
