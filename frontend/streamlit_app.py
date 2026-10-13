import os



import re



import streamlit as st



from app.document_loader import load_pdf



from app.chunker import create_chunks



from app.vector_store import build_vector_database



from app.document_library import load_documents, add_document



from app.document_manager import create_document_info



from app.rag_pipeline import run_rag



from app.evidence_verifier import verify_answer



from app.cross_document_comparator import compare_documents



# ============================================================



# PAGE CONFIG



# ============================================================



st.set_page_config(



    page_title="DocuTrace",



    page_icon="🔎",



    layout="wide"



)



# ============================================================



# CONSTANTS



# ============================================================



DATA_DIR = "data"



os.makedirs(DATA_DIR, exist_ok=True)



STOP_WORDS = {



    "the", "is", "are", "was", "were", "a", "an",



    "and", "or", "of", "to", "in", "on", "for",



    "with", "from", "by", "as", "at", "this",



    "that", "these", "those", "it", "its", "be",



    "can", "may", "will", "which", "what", "how",



    "why", "where", "when", "who", "does", "do",



    "about", "into", "than", "then", "their",



    "there", "they", "them", "also"



}



# ============================================================



# DOCUMENT INTELLIGENCE HELPERS



# ============================================================



def clean_words(text):



    words = re.findall(



        r"[A-Za-z0-9]+",



        text.lower()



    )



    return [



        word



        for word in words



        if word not in STOP_WORDS



        and len(word) > 2



    ]



def extract_keywords(text, limit=10):



    words = clean_words(text)



    frequency = {}



    for word in words:



        frequency[word] = (



            frequency.get(word, 0) + 1



        )



    sorted_words = sorted(



        frequency.items(),



        key=lambda item: item[1],



        reverse=True



    )



    return [



        word



        for word, count in sorted_words[:limit]



    ]



def find_keyword_pages(pages, keywords):



    keyword_pages = {}



    for keyword in keywords:



        matches = []



        for page in pages:



            text = page.get(



                "text",



                ""



            ).lower()



            if keyword.lower() in text:



                matches.append(



                    page.get(



                        "page",



                        "Unknown"



                    )



                )



        keyword_pages[keyword] = matches



    return keyword_pages



def detect_headings(pages):



    headings = []



    for page in pages:



        text = page.get(



            "text",



            ""



        )



        lines = text.splitlines()



        for line in lines:



            line = line.strip()



            if not line:



                continue



            if len(line) > 100:



                continue



            words = line.split()



            if len(words) > 12:



                continue



            if (



                line.isupper()



                or line.endswith(":")



                or (



                    len(words) <= 8



                    and line[0].isupper()



                )



            ):



                headings.append({



                    "text": line,



                    "page": page.get(



                        "page",



                        "Unknown"



                    )



                })



    return headings[:30]



def sentence_score(sentence, keywords):



    words = set(



        clean_words(sentence)



    )



    overlap = words.intersection(



        set(keywords)



    )



    return len(overlap)



def extractive_summary(text, keywords, max_sentences=3):



    sentences = re.split(



        r"(?<=[.!?])\s+",



        text



    )



    scored = []



    for index, sentence in enumerate(sentences):



        sentence = sentence.strip()



        if len(sentence) < 30:



            continue



        score = sentence_score(



            sentence,



            keywords



        )



        scored.append(



            (



                score,



                index,



                sentence



            )



        )



    scored.sort(



        key=lambda item: (



            item[0],



            -item[1]



        ),



        reverse=True



    )



    selected = sorted(



        scored[:max_sentences],



        key=lambda item: item[1]



    )



    return [



        item[2]



        for item in selected



    ]



def analyze_single_document(



    file_path,



    filename



):



    try:



        pages = load_pdf(



            file_path



        )



        if not pages:



            return {



                "filename": filename,



                "pages": 0,



                "keywords": [],



                "headings": [],



                "summary": []



            }



        full_text = "\n".join(



            page.get(



                "text",



                ""



            )



            for page in pages



        )



        keywords = extract_keywords(



            full_text



        )



        headings = detect_headings(



            pages



        )



        summary = extractive_summary(



            full_text,



            keywords



        )



        keyword_pages = find_keyword_pages(



            pages,



            keywords



        )



        return {



            "filename": filename,



            "pages": len(pages),



            "keywords": keywords,



            "headings": headings,



            "summary": summary,



            "keyword_pages": keyword_pages



        }



    except Exception as e:



        return {



            "filename": filename,



            "pages": 0,



            "keywords": [],



            "headings": [],



            "summary": [],



            "error": str(e)



        }



def generate_fast_document_intelligence(



    documents



):



    results = []



    for document in documents:



        filename = document.get(



            "name"



        )



        file_path = document.get(



            "path"



        )



        if not file_path:



            continue



        result = analyze_single_document(



            file_path,



            filename



        )



        results.append(



            result



        )



    return results



def display_document_intelligence(



    results



):



    for result in results:



        filename = result.get(



            "filename",



            "Unknown"



        )



        st.markdown(



            f"#### 📄 {filename}"



        )



        if result.get("error"):



            st.error(



                result["error"]



            )



            continue



        col1, col2 = st.columns(2)



        with col1:



            st.metric(



                "Pages",



                result.get(



                    "pages",



                    0



                )



            )



        with col2:



            st.metric(



                "Keywords",



                len(



                    result.get(



                        "keywords",



                        []



                    )



                )



            )



        summary = result.get(



            "summary",



            []



        )



        if summary:



            st.markdown(



                "**Summary**"



            )



            for sentence in summary:



                st.write(



                    f"• {sentence}"



                )



        keywords = result.get(



            "keywords",



            []



        )



        if keywords:



            st.markdown(



                "**Key Topics**"



            )



            st.write(



                ", ".join(keywords)



            )



        headings = result.get(



            "headings",



            []



        )



        if headings:



            with st.expander(



                "View detected headings"



            ):



                for heading in headings:



                    st.write(



                        f"• {heading['text']} "



                        f"(Page {heading['page']})"



                    )



        st.divider()



# ============================================================



# CITATION HELPERS



# ============================================================



def extract_citation_ids(answer):



    if not answer:



        return []



    
    citation_ids = re.findall(
        r"\[(E\d+)\]",
        answer
    )




    # Remove duplicates while preserving order



    seen = set()



    unique_ids = []



    for citation_id in citation_ids:



        if citation_id in seen:



            continue



        seen.add(



            citation_id



        )



        unique_ids.append(



            citation_id



        )



    return unique_ids



def display_cited_evidence(



    answer,



    evidence



):



    if not evidence:



        return



    citation_ids = extract_citation_ids(



        answer



    )



    if not citation_ids:



        st.info(



            "No specific evidence citations were generated."



        )



        return



    evidence_lookup = {



        item["id"]: item



        for item in evidence



    }



    st.markdown(



        "### 🔎 Supporting Evidence"



    )



    for citation_id in citation_ids:



        item = evidence_lookup.get(



            citation_id



        )



        if not item:



            continue



        document = item.get(



            "document",



            "Unknown document"



        )



        page = item.get(



            "page",



            "Unknown"



        )



        relevance = item.get(



            "relevance"



        )



        st.markdown(



            f"#### 📌 [{citation_id}] "



            f"{document} — Page {page}"



        )



        with st.expander(



            "View supporting passage"



        ):



            st.write(



                item.get(



                    "text",



                    ""



                )



            )



            if relevance is not None:



                st.caption(



                    f"Retrieval relevance: "



                    f"{relevance}"



                )



        st.markdown("---")



# ============================================================



# EVIDENCE VERIFICATION UI



# ============================================================



def display_evidence_verification(



    answer,



    evidence



):



    if not answer or not evidence:



        return



    citation_ids = extract_citation_ids(



        answer



    )



    if not citation_ids:



        return



    st.markdown(



        "### 🔎 Evidence Verification"



    )



    with st.spinner(



        "Verifying claims against document evidence..."



    ):



        try:



            verification_results = verify_answer(



                answer,



                evidence



            )



        
        except Exception as e:
            import traceback
            traceback.print_exc()
            st.error(
                f"Evidence verification failed: {e}"
            )
            return

    if not verification_results:
        st.info(
            "No cited claims were available for verification."
        )




    if not verification_results:



        st.info(



            "No cited claims were available for verification."



        )



        return



    supported_count = 0



    partial_count = 0



    unsupported_count = 0



    for result in verification_results:



        status = result.get(



            "status",



            "Unsupported"



        )



        if status == "Supported":



            icon = "✅"



            supported_count += 1



        elif status == "Partially Supported":



            icon = "⚠️"



            partial_count += 1



        else:



            icon = "❌"



            unsupported_count += 1



        st.markdown(



            f"#### {icon} {status}"



        )



        st.write(



            result.get(



                "claim",



                ""



            )



        )



        evidence_ids = result.get(



            "evidence_ids",



            []



        )



        if evidence_ids:



            st.caption(



                "Evidence: "



                + ", ".join(



                    evidence_ids



                )



            )



        reason = result.get(



            "reason",



            ""



        )



        if reason:



            st.caption(



                f"Reason: {reason}"



            )



        st.markdown("---")



    # Verification summary



    st.markdown(



        "**Verification Summary**"



    )



    col1, col2, col3 = st.columns(3)



    with col1:



        st.metric(



            "Supported",



            supported_count



        )



    with col2:



        st.metric(



            "Partially Supported",



            partial_count



        )



    with col3:



        st.metric(



            "Unsupported",



            unsupported_count



        )



# ============================================================



# HEADER



# ============================================================



st.title(



    "🔎 DocuTrace"



)



st.write(



    "Upload documents, search across them, "



    "and get evidence-backed answers."



)



# ============================================================

# UPLOAD SECTION

# ============================================================



st.header("Upload a PDF")



uploaded_file = st.file_uploader(

    "Upload a PDF",

    type=["pdf"]

)



if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    filename = uploaded_file.name

    save_path = os.path.join(DATA_DIR, filename)



    try:

        # Always save the bytes currently uploaded. This avoids indexing

        # stale contents when a file with the same name already exists.

        with open(save_path, "wb") as f:

            f.write(file_bytes)



        # Keep the filename as the document ID because the document

        # selector, retriever, comparator, and reindex script use names.

        document_id = filename



        pages = load_pdf(save_path)

        chunks = create_chunks(pages)



        # Replace existing chunks for this filename before indexing.

        # Import locally so this also works if the vector store module

        # was updated to expose delete_document().

        from app.vector_store import delete_document

        delete_document(document_id)



        build_vector_database(

            chunks,

            document_id=document_id

        )



        document = {

            "name": filename,

            "path": save_path,

            "document_id": document_id

        }



        add_document(document)



        st.success(

            f"{filename} uploaded and indexed successfully."

        )



    except Exception as e:

        st.error(

            f"Failed to process document: {e}"

        )





# ============================================================



# DOCUMENT LIBRARY



# ============================================================



st.header(



    "📚 Your Documents"



)



documents = load_documents()



if documents:



    for document in documents:



        st.write(



            f"- {document.get('name', 'Unknown')}"



        )



else:



    st.info(



        "No documents uploaded yet."



    )



# ============================================================



# DOCUMENT SELECTION



# ============================================================



document_names = [



    document.get(



        "name"



    )



    for document in documents



]



st.header(



    "Select documents to query:"



)



st.write(



    "Choose one or more documents:"



)



selected_documents = st.multiselect(



    "Documents",



    document_names,



    label_visibility="collapsed"



)



# ============================================================



# DOCUMENT INTELLIGENCE



# ============================================================



st.header(



    "📄 Document Intelligence"



)



st.write(



    "Select documents to analyze:"



)



intelligence_documents = st.multiselect(



    "Documents to analyze",



    document_names,



    key="intelligence_documents",



    label_visibility="collapsed"



)



if st.button(



    "Analyze Selected Documents"



):



    if not intelligence_documents:



        st.warning(



            "Please select at least one document."



        )



    else:



        selected_info = [



            document



            for document in documents



            if document.get("name")



            in intelligence_documents



        ]



        with st.spinner(



            "Analyzing documents..."



        ):



            intelligence_results = (



                generate_fast_document_intelligence(



                    selected_info



                )



            )



        display_document_intelligence(



            intelligence_results



        )



# ============================================================



# ASK QUESTION



# ============================================================



st.header(



    "Ask a question about your documents:"



)



query = st.text_input(



    "Question",



    placeholder=(



        "Example: What is packet switching?"



    ),



    label_visibility="collapsed"



)



if st.button(



    "Ask"



):



    if not query.strip():



        st.warning(



            "Please enter a question."



        )



    elif not selected_documents:



        st.warning(



            "Please select at least one document."



        )



    else:



        with st.spinner(



            "Searching documents and generating answer..."



        ):



            try:



                answer, evidence = run_rag(



                    query,



                    document_ids=selected_documents



                )



                st.markdown(



                    "### Answer"



                )



                st.markdown(



                    answer



                )



                # ------------------------------------------------



                # Supporting evidence



                # ------------------------------------------------



                display_cited_evidence(



                    answer,



                    evidence



                )



                # ------------------------------------------------



                # Evidence verification



                # ------------------------------------------------



                try:
                    display_evidence_verification(
                        answer,
                        evidence
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
                    st.error("Verification error. Check the PowerShell terminal.")



            except Exception as e:



                st.error(



                    f"An error occurred: {e}"



                )



# ============================================================



# CROSS-DOCUMENT COMPARISON



# ============================================================



st.header("🔀 Compare Documents")



st.write(



    "Compare information across two or more documents using "



    "retrieved evidence and citations."



)



comparison_document_names = st.multiselect(



    "Select documents to compare",



    document_names,



    key="comparison_documents"



)



comparison_query = st.text_input(



    "Comparison question",



    placeholder="Example: Compare the OSI model and TCP/IP model.",



    key="comparison_query"



)



if st.button("Compare Documents", key="run_document_comparison"):



    if len(comparison_document_names) < 2:



        st.warning("Please select at least two documents to compare.")



    elif not comparison_query.strip():



        st.warning("Please enter a comparison question.")



    else:



        with st.spinner("Retrieving evidence and comparing documents..."):



            try:



                comparison, comparison_evidence = compare_documents(



                    query=comparison_query,



                    document_ids=comparison_document_names



                )





                st.markdown("### 🔎 Comparison")



                st.markdown(comparison)



                st.markdown("### 📚 Evidence by Document")



                evidence_by_document = {}



                for item in comparison_evidence:

                    document_name = (

                        item.get("document")

                        or item.get("source")

                        or item.get("document_id")

                        or "Unknown document"

                    )



                    evidence_by_document.setdefault(

                        document_name, []

                    ).append(item)



                for selected_name in comparison_document_names:



                    matching_evidence = evidence_by_document.get(selected_name, [])



                    st.markdown(f"#### 📄 {selected_name}")



                    if not matching_evidence:



                        st.warning(



                            "No relevant evidence was retrieved from this document "



                            "for the comparison question."



                        )



                        continue



                    for item in matching_evidence:



                        evidence_id = item.get("id", "Unknown")



                        page = item.get("page", "Unknown")



                        relevance = item.get("relevance")



                        with st.expander(



                            f"[{evidence_id}] Page {page}"



                        ):



                            st.write(item.get("text", ""))



                            if relevance is not None:



                                st.caption(



                                    f"Retrieval relevance: {relevance}"



                                )



            except Exception as e:



                st.error(f"Document comparison failed: {e}")
