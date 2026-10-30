# Local Document Q&A Engine

> Ask questions about any PDF — fully local, zero data leaves your machine.
> Every question is answered twice, with and without RAG, so you can see what retrieval adds.

Built with a custom HNSW vector index in C++, Ollama for local LLM inference, Flask REST API, and a React frontend.

---

## Architecture

```
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
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector Index | Custom HNSW in C++ (from scratch) |
| Embeddings | nomic-embed-text via Ollama (768D, local) |
| LLM | llama3.2 via Ollama (fully local) |
| Backend | Python, Flask, PyMuPDF |
| Frontend | React, Vite |

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
5. Those chunks become numbered sources for `llama3.2`, which streams an answer citing them
6. In parallel, the same model answers the same question with no document, as the baseline

The model never sees the full document — only the most relevant chunks. This keeps answers accurate and fast.

## RAG vs no RAG

Click **Compare** and the page shows:

- **Without RAG**: llama3.2 on its own. For anything specific to your PDF it will guess or refuse.
- **With RAG**: llama3.2 given the top 5 chunks, citing them as [1], [2]...
- **Retrieval**: the chunks used, with page numbers and similarity, plus HNSW vs brute-force search time,
  how many vectors each compared, and how many of HNSW's results match the exact top 5.

Ask questions only your document can answer (names, numbers, definitions) to see the difference most clearly.

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

### 2. Compile the HNSW engine (output must sit in `hnsw/`)
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

Optional environment variables: `OLLAMA_URL`, `EMBED_MODEL`, `LLM_MODEL`, `HNSW_EXE`, `FLASK_DEBUG=1`.
The frontend reads `VITE_API_URL` (default `http://127.0.0.1:5000`).

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`, upload a PDF, start asking questions.

---

## Project Structure
```
rag-engine/
├── hnsw/
│   ├── hnsw.cpp          # HNSW vector index (C++, from scratch)
│   ├── hnsw.exe          # compiled binary (not committed)
│   ├── index.bin         # HNSW graph + vectors (generated)
│   └── chunks.json       # chunk text and page numbers (generated)
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
```

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

