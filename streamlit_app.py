import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
#from langchain.text_splitters import RecursiveCharacterTextSplitter  # Corrected import path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Dict, Any, Optional

# Streamlit configuration
st.set_page_config(
    page_title="Ollama PDF RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment variable for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


@st.cache_resource(show_spinner=True)
def extract_model_names(models_info: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Extract available model names from the models information."""
    return [model["name"] for model in models_info["models"]]


def create_vector_db(file_upload: st.file_uploader) -> Chroma:
    """Create a vector database from a PDF file."""
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
    loader = UnstructuredPDFLoader(path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)  # Correct usage
    chunks = text_splitter.split_documents(data)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    shutil.rmtree(temp_dir)
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """Process user question using the vector database and selected model."""
    llm = ChatOllama(model=selected_model)
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        PromptTemplate(
            input_variables=["question"],
            template="""Generate 2 different versions of the question for multi-perspective document retrieval.
            Original question: {question}""",
        ),
    )
    template = """Answer based on context:
    {context}
    Question: {question}"""
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_template(template)
        | llm
        | StrOutputParser()
    )
    return chain.invoke(question)


@st.cache_data
def extract_all_pages_as_images(file_upload: st.file_uploader) -> List[Any]:
    """Extract all pages from a PDF file as images."""
    with pdfplumber.open(file_upload) as pdf:
        return [page.to_image().original for page in pdf.pages]


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """Delete the vector database and clear session state."""
    if vector_db:
        vector_db.delete_collection()
        st.session_state.clear()
        st.success("Collection deleted successfully.")
        st.rerun()


def main():
    """Main Streamlit application."""
    st.subheader("🧠 Ollama PDF RAG Playground", divider="gray")

    models_info = ollama.list()
    available_models = extract_model_names(models_info)
    selected_model = st.selectbox("Choose a model", available_models)

    # File upload or sample PDF
    file_upload = st.file_uploader("Upload a PDF", type="pdf")
    use_sample = st.checkbox("Use sample PDF")

    if use_sample:
        sample_path = "scammer-agent.pdf"
        if os.path.exists(sample_path):
            with st.spinner("Processing sample PDF..."):
                loader = UnstructuredPDFLoader(sample_path)
                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=7500, chunk_overlap=100
                ).split_documents(loader.load())
                vector_db = Chroma.from_documents(
                    chunks, OllamaEmbeddings("nomic-embed-text"), "myRAG"
                )
                st.session_state["vector_db"] = vector_db
                st.session_state["pdf_pages"] = extract_all_pages_as_images(
                    open(sample_path, "rb")
                )
    elif file_upload:
        with st.spinner("Processing uploaded PDF..."):
            st.session_state["vector_db"] = create_vector_db(file_upload)
            st.session_state["pdf_pages"] = extract_all_pages_as_images(file_upload)

    # Display PDF
    if "pdf_pages" in st.session_state:
        zoom = st.slider("Zoom Level", 100, 1000, 700, 50)
        for page in st.session_state["pdf_pages"]:
            st.image(page, width=zoom)

    # Chat interface
    vector_db = st.session_state.get("vector_db")
    if vector_db:
        prompt = st.chat_input("Enter your question...")
        if prompt:
            response = process_question(prompt, vector_db, selected_model)
            st.chat_message("assistant", avatar="🤖").markdown(response)

    # Clear vector DB
    if st.button("Delete vector DB"):
        delete_vector_db(vector_db)


if __name__ == "__main__":
    main()
