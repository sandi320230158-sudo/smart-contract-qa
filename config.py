"""
Central configuration for Smart Contract Q&A.
Edit these values to tune the application.
"""

CONFIG = {
    # --- chunking ---
    "chunk_size":          500,   # characters per chunk
    "chunk_overlap":       100,   # overlap between chunks

    # --- retrieval ---
    "top_k":                 6,   # number of chunks returned per query

    # --- models ---
    "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model":   "google/flan-t5-large",

    # --- generation ---
    "max_new_tokens":      512,

    # --- guard-rail ---
    "relevance_threshold": 0.10,  # minimum cosine similarity to accept a query
}
