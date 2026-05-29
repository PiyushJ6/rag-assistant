from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
import shutil
import os

CHROMA_PATH = "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
def test_loading(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Number of pages loaded: {len(documents)}")
    print(f"Type of first item: {type(documents[0])}")
    print(f"\nFirst page content (first 300 chars):")
    print(documents[0].page_content[:300])
    print(f"\nMetadata of first page:")
    print(documents[0].metadata)
    return documents

def test_splitting(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Pages before splitting: {len(documents)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Chunks after splitting: {len(chunks)}")
    print(f"\n--- Chunk 1 ---")
    print(chunks[0].page_content)
    print(f"\n--- Chunk 2 ---")
    print(chunks[1].page_content)
    print(f"\n--- Chunk 3 ---")
    print(chunks[2].page_content)
    print(f"\nNotice: end of chunk 1 and start of chunk 2 share some text (that's the overlap)")
    return chunks

def test_embeddings():
    embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    sentences = [
        "The Transformer model uses attention mechanisms",
        "Attention allows the model to focus on relevant parts of the input",
        "I like eating pizza on Sundays"
    ]
    vectors = embed_model.embed_documents(sentences)
    print(f"Each sentence becomes a vector of {len(vectors[0])} numbers")
    print(f"\nFirst 5 numbers of sentence 1: {vectors[0][:5]}")
    print(f"First 5 numbers of sentence 2: {vectors[1][:5]}")
    print(f"First 5 numbers of sentence 3: {vectors[2][:5]}")
    def cosine_sim(a, b):
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sim_1_2 = cosine_sim(vectors[0], vectors[1])
    sim_1_3 = cosine_sim(vectors[0], vectors[2])
    print(f"\nSimilarity between sentence 1 and 2 (both about attention): {sim_1_2:.3f}")
    print(f"Similarity between sentence 1 and 3 (unrelated):            {sim_1_3:.3f}")
    print(f"\nHigher = more related. This is how retrieval finds relevant chunks.")

def load_and_index(pdf_path: str) -> Chroma:
    """Full pipeline: PDF → chunks → embeddings → ChromaDB"""

    # Always wipe old DB first — prevents duplicate chunks
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("  Cleared old ChromaDB")

    print("Stage 1: Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    print("Stage 2: Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Stage 3: Embedding and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"  Done. Saved to '{CHROMA_PATH}' folder")
    return vectorstore

def load_existing() -> Chroma:
    """Load already-indexed vectorstore from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )