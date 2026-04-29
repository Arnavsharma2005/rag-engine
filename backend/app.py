import os
import subprocess
import requests
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HNSW_EXE = r"C:\Users\arnav\rag-engine\hnsw\hnsw.exe"
HNSW_CWD = r"C:\Users\arnav\rag-engine\hnsw"
OLLAMA_URL = "http://localhost:11434"

# ---------- helpers ----------

def get_embedding(text: str) -> list[float]:
    """Convert text to a 768D vector using nomic-embed-text via Ollama."""
    r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
        "model": "nomic-embed-text",
        "prompt": text
    })
    return r.json()["embedding"]

def vec_to_str(vec: list[float]) -> str:
    """Serialize vector to comma-separated string for C++ subprocess."""
    return ",".join(f"{v:.6f}" for v in vec)

def hnsw_insert(text: str, vec: list[float]):
    """Insert a chunk into the HNSW index."""
    subprocess.run(
        [HNSW_EXE, "insert", text, vec_to_str(vec)],
        cwd=HNSW_CWD, capture_output=True
    )

def hnsw_search(vec: list[float]) -> list[str]:
    """Search HNSW index, return top-5 matching chunk texts."""
    result = subprocess.run(
        [HNSW_EXE, "search", vec_to_str(vec)],
        cwd=HNSW_CWD, capture_output=True, text=True
    )
    chunks = []
    for line in result.stdout.strip().split("\n"):
        if "\t" in line:
            _, text = line.split("\t", 1)
            chunks.append(text)
    return chunks

def chunk_text(text: str, size=300, overlap=50) -> list[str]:
    """Split text into overlapping chunks of ~300 words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)
        i += size - overlap
    return chunks

# ---------- routes ----------

@app.route("/upload", methods=["POST"])
def upload():
    """Upload a PDF, extract text, chunk it, embed each chunk, insert into HNSW."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in pdf:
        full_text += page.get_text()

    # Reset index for new document
    subprocess.run([HNSW_EXE, "reset"], cwd=HNSW_CWD, capture_output=True)

    chunks = chunk_text(full_text)
    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        vec = get_embedding(chunk)
        hnsw_insert(chunk, vec)

    return jsonify({"status": "ok", "chunks": len(chunks)})


@app.route("/ask", methods=["POST"])
def ask():
    """Take a question, find relevant chunks, stream answer from llama3.2."""
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "no question"}), 400

    # Embed the question and find closest chunks
    q_vec = get_embedding(question)
    chunks = hnsw_search(q_vec)
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on this document."

Context:
{context}

Question: {question}
Answer:"""

    def generate():
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": True
        }, stream=True)
        for line in r.iter_lines():
            if line:
                import json
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]

    return Response(stream_with_context(generate()), mimetype="text/plain")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)