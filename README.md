
<div align="center">

---

## ✨ Key Features

- 📜 **Multilingual Transcript Processing**: Automatically fetches timestamped transcripts for English (`en`) and Arabic (`ar`) videos via `youtube-transcript-api`.
- 🧠 **Smart Sentence-Boundary Chunking**: Groups transcript snippets dynamically to maintain full semantic sentences and prevent contextual fragmentation.
- 🎯 **Fast Dense Retrieval (FAISS)**: Generates embeddings using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and indexes them in a local FAISS vector store.
- 📝 **Hierarchical Map-Reduce Summarization**: Synthesizes lengthy transcripts in parallel batches without exceeding LLM context windows or losing critical details.
- ⏱️ **Timestamped Ground-Truth Citations**: Retrieves relevant video passages and appends exact video timestamps (`[start-end]`) to ensure hallucination-free answers.
- 🔒 **Secure Enterprise Architecture**: Features constant-time Bearer token authentication (`secrets.compare_digest`), Pydantic v2 data validation[cite: 5, 6], and thread-safe async locks.

---

## 🏗️ Architecture

```text
┌──────────────────────────┐
│  YouTube Video URL       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐     ┌────────────────────────────────┐
│ Transcript Extraction    ├────►│ Smart Chunking & FAISS Vector  │
│ (English & Arabic)       │     │ Store Generation               │
└──────────────────────────┘     └───────────────┬────────────────┘
                                                 │
                                                 ▼
┌──────────────────────────┐     ┌────────────────────────────────┐
│ Streamlit Frontend       ├────►│ FastAPI RAG Backend            │
│ Interactive UI           │     │ (Gemini 3.8 Flash + Embeddings)│
└──────────────────────────┘     └───────────────┬────────────────┘
                                                 │
                                                 ▼
                                 ┌────────────────────────────────┐
                                 │ Grounded Answers with          │
                                 │ Verifiable Timestamp Citations │
                                 └────────────────────────────────┘
```
