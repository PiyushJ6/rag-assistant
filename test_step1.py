from rag.loader import load_and_index, load_existing, CHROMA_PATH
from rag.chain import answer_question
import os

# Create embeddings only if DB doesn't exist
if not os.path.exists(CHROMA_PATH):
    load_and_index("testing.pdf")

# Load existing vector DB
vectorstore = load_existing()

questions = [
    "What optimizer was used for training the Transformer?",
    "How does multi-head attention work?",
    "What were the BLEU scores achieved on the translation tasks?"
]

for question in questions:
    print("=" * 60)
    print(f"Question: {question}")
    print("=" * 60)
    result = answer_question(question, vectorstore)
    print(f"\nAnswer:\n{result['answer']}")
    print()