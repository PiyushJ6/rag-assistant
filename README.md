# Document Q&A Assistant

An intelligent RAG (Retrieval Augmented Generation) system that answers questions 
grounded in uploaded PDF documents.

## How it works
1. PDF is loaded and split into 800-character chunks with 150-character overlap
2. Chunks are embedded using sentence-transformers/all-MiniLM-L12-v2
3. Embeddings stored in ChromaDB vector database
4. User questions are embedded and matched via MMR retrieval
5. Top chunks sent to LLaMA 3.1 8B (Groq API) for answer generation
6. Answer returned with source page citations

## Stack
- LangChain + LangChain-Groq
- ChromaDB vector database  
- HuggingFace sentence-transformers
- Groq API (LLaMA 3.1 8B)
- Streamlit UI

## Run locally
pip install -r requirements.txt
# Add GROQ_API_KEY to .env
streamlit run app.py
