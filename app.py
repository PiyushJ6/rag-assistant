import streamlit as st
import tempfile
import os
from rag.loader import load_and_index, load_existing, CHROMA_PATH
from rag.chain import answer_question

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Q&A Assistant")
st.caption("Upload a PDF and ask anything about it — answers grounded in your document.")

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        if st.session_state.get("indexed_file") != uploaded_file.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Indexing '{uploaded_file.name}'..."):
                st.session_state.vectorstore = load_and_index(tmp_path)
                st.session_state.indexed_file = uploaded_file.name
                st.session_state.messages = []
            os.unlink(tmp_path)
            st.success(f"✅ Ready! Ask anything about this document.")
        else:
            st.success(f"✅ '{uploaded_file.name}' already indexed.")

    st.divider()
    st.markdown("**Model:** LLaMA 3.1 8B (Groq)")
    st.markdown("**Embeddings:** MiniLM-L12-v2")
    st.markdown("**Vector DB:** ChromaDB")
    st.markdown("**Retrieval:** MMR")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main chat area ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.info("👈 Upload a PDF from the sidebar to get started.")
    st.stop()

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 View source chunks"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', '?')}**")
                    st.text(doc.page_content[:400])
                    st.divider()

# Single chat input
if question := st.chat_input("Ask a question about your document...", key="main_input"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(question, st.session_state.vectorstore)

        answer = result.get("answer", "No answer returned")
        st.write(answer)

        if result.get("sources"):
            with st.expander("📎 View source chunks"):
                for i, doc in enumerate(result["sources"]):
                    st.markdown(f"**Chunk {i+1} — Page {doc.metadata.get('page', '?')}**")
                    st.text(doc.page_content[:400])
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", [])
    })