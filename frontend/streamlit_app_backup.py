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
