\# 🔎 DocuTrace



\### Search. Compare. Trace the evidence.



DocuTrace is an evidence-backed document search and comparison system that allows users to upload PDF documents, ask questions, retrieve relevant passages, and compare information across documents with page-level evidence citations.



The project combines document processing, semantic search, retrieval-augmented generation (RAG), local language models, and evidence verification into a single Streamlit application.



\---



\## ✨ Features



\- 📄 Upload and process PDF documents

\- 🔎 Semantic search across document content

\- 💬 Ask questions about uploaded documents

\- 🧠 Retrieval-Augmented Generation (RAG)

\- 📑 Retrieve relevant evidence passages

\- 📍 Page-level document citations

\- 🔄 Compare information across multiple documents

\- ⚖️ Identify differences and similarities using retrieved evidence

\- 🧾 Display supporting evidence for generated answers

\- 🔐 Local LLM inference using Ollama

\- 🗂️ Document management and indexing

\- ✅ Evidence and citation verification

\- 🧪 Automated tests for major application components



\---



\## 🏗️ How DocuTrace Works



```text

&#x20;               ┌─────────────────────┐

&#x20;               │     PDF Documents   │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │  Document Loader    │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Text Chunking       │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Embeddings          │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Vector Store        │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Semantic Retrieval  │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ RAG Pipeline        │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Local LLM (Ollama)  │

&#x20;               └──────────┬──────────┘

&#x20;                          │

&#x20;                          ▼

&#x20;               ┌─────────────────────┐

&#x20;               │ Answer + Evidence   │

&#x20;               └─────────────────────┘

