# from src.utils import load_model, load_pdf, text_split, download_hugging_face_embeddings, make_db
# import dotenv
# import os
# from langchain.chains import RetrievalQA

# dotenv.load_dotenv()
# llm_key = os.getenv("llm_key")

# llm = load_model(llm_key)
# embeddings = download_hugging_face_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# data = load_pdf("pdfs")
# extracted_data = text_split(data)

# vectordb = make_db(embeddings, extracted_data)

# retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# qa_chain = RetrievalQA.from_chain_type(llm=llm,
#                                   chain_type="stuff",
#                                   retriever=retriever,
#                                   return_source_documents=True)

# query = "what are some professinal ethics"

# def process_llm_response(llm_response):
#     print(llm_response['result'])
#     print('\n\nSources:')
#     for source in llm_response["source_documents"]:
#         print(source.metadata['source'])

# llm_response = qa_chain(query)



import streamlit as st
from src.utils import load_model, load_pdf, text_split, download_hugging_face_embeddings, make_db
import dotenv
import os
from langchain.chains import RetrievalQA

# Load environment variables
dotenv.load_dotenv()
llm_key = os.getenv("llm_key")

# Ensure LLM API key is present
if not llm_key:
    st.error("Missing Hugging Face API Key in environment variables.")
    st.stop()

# Load models and embeddings
llm = load_model(llm_key)
embeddings = download_hugging_face_embeddings("sentence-transformers/all-MiniLM-L6-v2")

# Streamlit UI Components
st.title("Document Query with LLM-Powered Retrieval")
st.sidebar.header("Upload and Query PDFs")

# File upload section
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        # Save uploaded files locally
        for uploaded_file in uploaded_files:
            with open(f"pdfs/{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.read())

        # Load and process PDFs
        data = load_pdf("pdfs")
        extracted_data = text_split(data)

        # Create vector database
        vectordb = make_db(embeddings, extracted_data)
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})

        # Initialize RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        st.success("PDFs processed successfully!")

        # Query input
        query = st.text_input("Enter your query:")

        if query:
            with st.spinner("Fetching results..."):
                llm_response = qa_chain(query)

            # Display result
            st.subheader("Response")
            st.write(llm_response['result'])

            st.subheader("Sources")
            for source_doc in llm_response["source_documents"]:
                st.write(f"- {source_doc.metadata['source']}")
else:
    st.info("Please upload one or more PDF files.")
