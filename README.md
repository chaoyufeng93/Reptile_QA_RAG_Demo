# 🧠 Reptile RAG Agent (LangGraph + Hybrid Retrieval)

A domain-specific RAG system for **reptile husbandry and veterinary Q&A**,  
built with **LangChain + LangGraph + Hybrid Retrieval + Reranking**.

---

## 🚀 Features

### ✅ 1. Hybrid Retrieval (Core Highlight)
- Vector search (FAISS)
- Keyword search (BM25)
- Deduplication (overlap-based)
- CrossEncoder reranking

👉 Solves:
- Unstable embedding recall
- Weak keyword precision
- Noisy long-document chunks

---

### ✅ 2. LangGraph Agent Pipeline

```text
User (Chinese)
   ↓
Rewrite Node (CN → EN query)
   ↓
Retriever Node (Hybrid Retrieval + Rerank)
   ↓
Answer Node (Grounded Answer)
   ↓
Final Answer (Chinese)
```

---

### ✅ 3. Domain Optimization (Reptile)

- Query rewriting tailored for:
  - husbandry
  - veterinary
  - species-specific terminology
- Answer constraints:
  - No hallucination
  - Parameter-focused (temperature, humidity, etc.)
  - Conservative medical responses

---

### ✅ 4. Structured Document Pipeline

Supports:
- PDF → TXT
- Section-based splitting (via TOC)
- Metadata:
  - `bookname`
  - `section`

---

## 📂 Project Structure

```text
.
├── rag.py                # Vector store & retriever
├── agent_graph.py        # LangGraph pipeline
├── data_prep.py          # Data processing
├── settings.py           # Config (user-defined)
├── sources/
│   ├── pdf/
│   ├── txt/
│   └── faiss_index/
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- langchain
- langgraph
- faiss-cpu
- sentence-transformers
- pymupdf
- python-dotenv

---

## 🔑 Environment

```bash
export OPENAI_API_KEY=your_key
```

Or input at runtime.

---

## 📦 Usage

### 1️⃣ Build Vector Store

```bash
python rag.py
```

### 2️⃣ Run Agent

```bash
python agent_graph.py
```

Example:

```python
INPUT = "My hognose snake has been hiding for over a week. What could be the reason?"
```

---

## 🧩 Core Components

### 🔹 Document Processing
- PDF extraction
- Cleaning
- Section structuring

### 🔹 Chunking
- RecursiveCharacterTextSplitter
- Custom chunk size & overlap

### 🔹 Retrieval
- FAISS + BM25 hybrid
- Deduplication
- Reranking

### 🔹 Reranker
- CrossEncoder
- Improves semantic precision

### 🔹 LangGraph Nodes

- Rewrite Node → query optimization
- Retriever Node → retrieval + rerank
- Answer Node → grounded generation

---

## 🧠 Design Philosophy

> Retrieval is not just recall — it's filtering.

- RAG ≠ embedding search
- Agent = information flow controller

---

## 📈 Improvements

- Query routing
- Dynamic chunking
- Metadata filtering
- Multi-hop retrieval
- BM25 persistence

---

## 🎯 Positioning

- Lightweight but complete RAG system
- Strong foundation for vertical AI agents
- Easily extendable to other domains

---

## ⚠️ Limitations

- Depends on OpenAI API
- BM25 not persisted
- No async / streaming
- No evaluation pipeline

---

## 🧩 Future Direction

Can be extended to:
- Financial RAG systems
- News-driven agents
- Regime detection pipelines

---

## 📄 Source Files

Generated from:
- rag.py
- agent_graph.py
- data_prep.py
