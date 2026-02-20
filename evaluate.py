"""
Offline evaluation helper.

Usage:
    python evaluate.py --file path/to/contract.pdf

The script builds a vector store from the document and runs a set of
retrieval-based sanity checks. No LLM calls are made — only embedding
similarity search — so it is fast and free to run.

Add your own (question, keyword) pairs to TEST_QUESTIONS below.
"""

import argparse
import sys
from pathlib import Path

# Reuse helpers from the main app
from config import CONFIG
from app import (
    ingest_file,
    chunk_documents,
    build_vector_store,
)

TEST_QUESTIONS = [
    ("What is the title of this project?",           "smart"),
    ("What domain does this project belong to?",     "langchain"),
    ("What type of application is being built?",     "web"),
    # Add your own pairs here ↓
]


def run_eval(filepath: str):
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    file_bytes = path.read_bytes()
    print(f"\nLoading '{path.name}' …")
    docs = ingest_file(file_bytes, path.name)
    chunks = chunk_documents(docs)
    vs = build_vector_store(chunks)

    print(f"\n{'Question':<50} {'Keyword':<20} Result")
    print("-" * 80)

    hits = 0
    for question, keyword in TEST_QUESTIONS:
        results = vs.similarity_search(question, k=CONFIG["top_k"])
        combined = " ".join(d.page_content for d in results).lower()
        question_words = [w for w in question.lower().split() if len(w) > 3]
        hit = keyword.lower() in combined or any(w in combined for w in question_words)
        hits += hit
        status = "✅ YES" if hit else "❌ NO"
        print(f"{question[:48]:<50} {keyword[:18]:<20} {status}")

    total = len(TEST_QUESTIONS)
    print(f"\nRetrieval accuracy: {hits}/{total} = {hits / total * 100:.0f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy.")
    parser.add_argument("--file", required=True, help="Path to PDF or DOCX contract")
    args = parser.parse_args()
    run_eval(args.file)
