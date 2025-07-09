import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# ----------------------------
# App Title
# ----------------------------
st.title("📄🧠 RAG with LangChain & Ollama (PDF Q&A)")
st.markdown("Upload a PDF and ask questions using Retrieval-Augmented Generation")

# ----------------------------
# Sidebar model selector
# ----------------------------
model_name = st.sidebar.selectbox("Choose Ollama model", ["mistral", "gemma:2b", "llama2"])
st.sidebar.markdown("✅ Make sure the model is pulled using `ollama pull model_name`")

# ----------------------------
# PDF Upload
# ----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    # Loading and processing with progress feedback
    with st.spinner("📄 Loading PDF..."):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    with st.spinner("🔀 Splitting text..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

    with st.spinner("🧠 Generating embeddings..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

    with st.spinner("📚 Building vector store..."):
        vector_db = Chroma.from_documents(texts, embedding=embeddings)

    retriever = vector_db.as_retriever()

    try:
        llm = Ollama(model=model_name)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        st.success(f"✅ Model `{model_name}` is ready! Ask a question below.")
        query = st.text_input("💬 Ask a question based on the uploaded PDF:")

        if query:
            with st.spinner("🤖 Generating answer..."):
                answer = qa_chain.run(query)
                st.write("### 🤖 Answer:")
                st.success(answer)

    except Exception as e:
        st.error(f"❌ Error connecting to Ollama: {e}")
