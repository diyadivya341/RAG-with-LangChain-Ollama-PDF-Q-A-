
📄🧠 RAG with LangChain, Streamlit & Ollama (PDF Q&A)
This project allows users to upload a PDF and interactively ask questions based on its content using Retrieval-Augmented Generation (RAG). It combines LangChain's powerful document processing, HuggingFace embeddings, Chroma vector store, and Ollama's local LLMs — all wrapped in a user-friendly Streamlit interface.

📌 Project Summary
The goal is to simulate a chatbot that answers user questions by referencing content from an uploaded PDF. Using LangChain's components and Ollama's local models like mistral, the app converts PDF content into embeddings, indexes them into a vector database (Chroma), and uses similarity search + LLM generation to provide human-like answers.

📂 Dataset Details
Users upload any PDF file (e.g., reports, research papers, notes)

The PDF is parsed and split into chunks using LangChain

Embeddings are generated using HuggingFace transformers

Chunks are stored in Chroma, a fast vector store

🛠️ Technologies Used:

Frontend & Deployment

Streamlit – interactive web interface

NLP & LLM Framework

LangChain – for document loading, splitting, retrieval

Ollama – to run local LLMs (e.g., mistral, gemma)

Embeddings & Vector Store

HuggingFace Transformers – for sentence embeddings

Chroma – fast, persistent vector search

PDF Handling

PyPDF2 (via langchain_community) – for reading PDFs

🔄 Project Workflow

Upload PDF: User uploads any PDF file

Text Splitting: Split into ~1000-character chunks with overlap

Embeddings: Convert text chunks into vectors using MiniLM

Vector DB: Store vectors in Chroma for similarity search

Model Selection: User chooses Ollama model (mistral, etc.)

Answering: Input questions → retrieve relevant chunks → generate answers using the selected model

🧠 Key Features:

Upload and analyze any PDF

Automatic text chunking and embedding

Select your preferred local LLM (Ollama)

Ask questions in natural language

Realtime response from content inside the PDF

📈 Model Info

Uses HuggingFace MiniLM L3 v2 for lightweight fast embeddings

Supports Mistral, Gemma, or any other Ollama-supported LLM

Answers generated based on relevant context from the document

🚀 How to Run Locally:

1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/your-username/rag-pdf-qa.git
cd rag-pdf-qa

2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv ollama-env
.\ollama-env\Scripts\activate    # On Windows

3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

4. Pull Ollama Model
bash
Copy
Edit
ollama pull mistral

5. Run the App
bash
Copy
Edit

streamlit run app.py

📦 Requirements

Create a requirements.txt like:

txt
Copy
Edit
streamlit
langchain
langchain-community
chromadb
sentence-transformers
PyPDF2
tqdm



![Screenshot 2025-07-08 214227](https://github.com/user-attachments/assets/75bf00e3-3ecd-42ba-a903-619dcd24a687)


![Screenshot 2025-07-08 214244](https://github.com/user-attachments/assets/c9167fc5-38dc-425f-b513-cf5fcf583eff)

