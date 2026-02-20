 Smart Contract Q&A Assistant

A RAG-powered (Retrieval-Augmented Generation) chatbot that lets you **upload any PDF or DOCX contract and ask questions about it** — all running locally on CPU, no API keys required.



---

 Features

-  **Upload** PDF or DOCX contracts
-  **Chat** with your document using natural language
-  **Summarize** contracts with one click (map-reduce approach)
-  **Guard-rails** — prompt injection detection & relevance filtering
-  **Source citations** — every answer links back to the exact page

---

##  Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/smart-contract-qa.git
cd smart-contract-qa
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> First run downloads ~500 MB of models. They are cached locally afterwards.

### 4. Run the app
```bash
python app.py
```

Open the URL printed in the terminal (usually `http://127.0.0.1:7860`).

---

##  Project Structure

```
smart-contract-qa/
├── app.py            # Main application (ingestion, RAG chain, Gradio UI)
├── config.py         # Tunable settings (models, chunk size, thresholds)
├── evaluate.py       # Offline retrieval evaluation script
├── requirements.txt  # Python dependencies
└── README.md
```

---

##  Configuration

Edit `config.py` to change models or tuning parameters:

| Key | Default | Description |
|-----|---------|-------------|
| `chunk_size` | 500 | Characters per chunk |
| `chunk_overlap` | 100 | Overlap between chunks |
| `top_k` | 6 | Chunks retrieved per query |
| `embed_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `llm_model` | `flan-t5-large` | Text generation model |
| `max_new_tokens` | 512 | Max tokens generated |
| `relevance_threshold` | 0.10 | Min cosine similarity to accept query |

---

##  Evaluation

Run retrieval accuracy checks against your own document:

```bash
python evaluate.py --file path/to/contract.pdf
```

Add custom (question, keyword) pairs inside `evaluate.py` under `TEST_QUESTIONS`.

---

##  Tech Stack

| Layer | Library |
|-------|---------|
| Document parsing | PyMuPDF, python-docx |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (CPU) |
| LLM | `google/flan-t5-large` via HuggingFace Transformers |
| Orchestration | LangChain |
| UI | Gradio |

---

##  Run on Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/smart-contract-qa/blob/main/notebook.ipynb)

> Replace `YOUR_USERNAME` with your GitHub username after pushing.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
