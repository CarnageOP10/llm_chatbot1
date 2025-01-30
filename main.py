import streamlit as st
from src.utils import load_model, load_pdf, text_split, download_hugging_face_embeddings, make_db
import dotenv
import os
from langchain.chains import RetrievalQA
from src.prompts import prompt_template
from langchain.prompts import PromptTemplate

# Load environment variables
dotenv.load_dotenv()
llm_key = os.getenv("llm_key")

# Ensure LLM API key is present
if not llm_key:
    st.error("Missing Hugging Face API Key in environment variables.")
    st.stop()

# Load models and embeddings
st.write("Loading the LLM model...")
llm = load_model(llm_key)
if llm is None:
    st.error("Failed to load the LLM model. Please check the API key or model configuration.")
    st.stop()

embeddings = download_hugging_face_embeddings("sentence-transformers/all-MiniLM-L6-v2")

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

st.title("Document Query with LLM-Powered Retrieval")

# Load and process PDFs from the "pdfs" directory
with st.spinner("Loading and processing PDFs from the 'pdfs' directory..."):
    data = load_pdf("pdfs")
    extracted_data = text_split(data)

    # Create vector database
    vectordb = make_db(embeddings, extracted_data)
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True, 
        chain_type_kwargs=chain_type_kwargs
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

