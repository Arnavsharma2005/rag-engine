# Local Document Q&A Engine

> Ask questions about any PDF — fully local, zero data leaves your machine.

Built with a custom HNSW vector index in C++, Ollama for local LLM inference, Flask REST API, and a React frontend.

---

## Architecture

PDF Upload
│
▼
Text Extraction (PyMuPDF)
│
▼
Chunking (300 words, 50-word overlap)
│
▼
Embedding (Ollama: nomic-embed-text → 768D vectors)
│
▼
HNSW Index (C++ custom implementation)
│
├── index.bin (persisted to disk)
│
Query
│
▼
Embed Question → HNSW Search → Top 5 Chunks
│
▼
LLM (Ollama: llama3.2) → Streamed Answer
│
▼
React Frontend (real-time typewriter stream)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Index | Custom HNSW in C++ (from scratch) |
| Embeddings | nomic-embed-text via Ollama (768D, local) |
| LLM | llama3.2 via Ollama (fully local) |
| Backend | Python, Flask, PyMuPDF |
| Frontend | React, Vite |
| Deployment | Vercel (frontend), Render (backend) |

---

## What is HNSW?

HNSW (Hierarchical Navigable Small World) is the vector search algorithm used by Pinecone, Weaviate, and Qdrant. Instead of comparing your query against every stored vector (O(N)), it builds a multi-layered graph where:

- **Upper layers** have sparse long-range connections — fast coarse navigation
- **Lower layers** have dense short-range connections — precise local search

This gives **O(log N) search time**. I implemented it from scratch in C++ including:
- Cosine distance metric
- Randomized layer assignment (exponential distribution)
- Greedy beam search with configurable `ef` parameter
- Binary serialization to disk (`index.bin`)

---

## How RAG Works Here

1. PDF is split into overlapping 300-word chunks
2. Each chunk is embedded into a 768D vector using `nomic-embed-text`
3. Vectors are inserted into the HNSW index
4. At query time, the question is embedded and HNSW finds the 5 closest chunks
5. Those chunks become context for `llama3.2` which streams the answer

The model never sees the full document — only the most relevant chunks. This keeps answers accurate and fast.

---

## Setup

### Prerequisites
- [Ollama](https://ollama.com) installed and running
- Python 3.10+
- Node.js 18+
- g++ (MinGW on Windows)

### 1. Pull Ollama models
```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

### 2. Compile the HNSW engine
```bash
cd hnsw
g++ -O2 -o hnsw.exe hnsw.cpp   # Windows
# g++ -O2 -o hnsw hnsw.cpp     # Linux/Mac
```

### 3. Backend
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a PDF, start asking questions.

---

## Project Structure
rag-engine/
├── hnsw/
│   ├── hnsw.cpp          # HNSW vector index (C++, from scratch)
│   └── hnsw.exe          # compiled binary (not committed)
├── backend/
│   ├── app.py            # Flask API
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   └── App.jsx       # React UI
│   ├── index.html
│   └── package.json
├── .gitignore
└── README.md

---

## Key Parameters

| Parameter | Value | What it controls |
|---|---|---|
| `M` | 16 | Max neighbors per node per layer — higher = more accurate, more memory |
| `ef` | 50 | Search beam width — higher = more accurate, slower |
| Chunk size | 300 words | Larger = more context per chunk, less precise retrieval |
| Overlap | 50 words | Prevents information loss at chunk boundaries |

---

## Author

**Arnav Sharma**  
B.Tech ECE, BMS College of Engineering  
[github.com/Arnavsharma2005](https://github.com/Arnavsharma2005)

