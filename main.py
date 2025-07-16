import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import tempfile

st.set_page_config(page_title="PDF RAG Chat", layout="wide")
st.title("ask your manual")

# Upload PDF files
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        docs = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = PyPDFLoader(tmp_file.name)
                docs.extend(loader.load())

        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # Create embeddings
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(splits, embedding)

        # Load Ollama model
        llm = OllamaLLM(model="mistral")

        # RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )

        # Chat interface
        st.subheader("🧠 Ask something about your PDFs:")
        user_question = st.text_input("Your question:")

        if user_question:
            with st.spinner("Thinking..."):
                result = qa_chain.invoke({"query": user_question})
                st.markdown(f"**Answer:** {result['result']}")
