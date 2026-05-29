from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

PROMPT_TEMPLATE = """
You are an expert research assistant.

Answer the question using ONLY the provided context.

Rules:
1. Do NOT use outside knowledge
2. If the answer is not in the context, say: "I couldn't find this in the document."
3. Include important technical details, equations, and numerical values when relevant
4. Ignore unrelated tables or metrics
5. Do not combine unrelated results
6. Mention the source page(s)

Context:
{context}

Question:
{question}

Answer:
"""

def answer_question(question: str, vectorstore: Chroma) -> dict:
    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 4,
            "lambda_mult": 0.5
        }
    )
    retrieved_docs = retriever.invoke(question)

    # Build context string
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        page = doc.metadata.get("page", "?")
        context_parts.append(f"[Chunk {i+1}, Page {page}]:\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    # Call LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted = prompt.format_messages(context=context, question=question)
    response = llm.invoke(formatted)

    return {
        "answer": response.content,
        "sources": retrieved_docs
    }