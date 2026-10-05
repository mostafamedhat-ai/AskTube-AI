# 🎥🤖 AskTube-AI

<div align="center">

**An Enterprise-Grade, End-to-End RAG Pipeline for YouTube Video Summarization & Context-Aware Q&A**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Core-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

AskTube-AI transforms long YouTube video transcripts into interactive, semantic knowledge bases. Built with **FastAPI**, **LangChain**, and **Google Gemini**, it allows users to generate structured video summaries and ask precise questions with automated, verifiable **[start-end]** timestamp citations.

---

## ✨ Key Features

- 📜 **Multilingual Transcript Processing**: Automatically fetches timestamped transcripts for English (`en`) and Arabic (`ar`) videos via `youtube-transcript-api`.
- 🧠 **Smart Sentence-Boundary Chunking**: Groups transcript snippets dynamically to maintain full semantic sentences and prevent contextual fragmentation.
- 🎯 **Fast Dense Retrieval (FAISS)**: Generates embeddings using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and indexes them in a local FAISS vector store.
- 📝 **Hierarchical Map-Reduce Summarization**: Synthesizes lengthy transcripts in parallel batches without exceeding LLM context windows or losing critical details.
- ⏱️ **Timestamped Ground-Truth Citations**: Retrieves relevant video passages and appends exact video timestamps (`[start-end]`) to ensure hallucination-free answers.
- 🔒 **Secure Enterprise Architecture**: Features constant-time Bearer token authentication (`secrets.compare_digest`), Pydantic v2 data validation, and thread-safe async locks.

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
