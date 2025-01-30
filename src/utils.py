from langchain import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def load_model(llm_key):
    llm = HuggingFaceHub(
    huggingfacehub_api_token=llm_key,
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.7, "max_length": 200}
    )


def download_hugging_face_embeddings(model_name):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()

    return documents


def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


def make_db(embeddings, splits):
    persist_directory = 'db'
    vectordb = Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory=persist_directory)
    vectordb.persist()
    vectordb = None
    vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
    return vectordb