import os
import json
import time
import threading
import subprocess
import requests
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# Only the local frontend may call the API, so other websites can't read your document
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HNSW_CWD = os.environ.get("HNSW_DIR", os.path.join(ROOT, "hnsw"))
HNSW_EXE = os.environ.get(
    "HNSW_EXE", os.path.join(HNSW_CWD, "hnsw.exe" if os.name == "nt" else "hnsw")
)
CHUNKS_FILE = os.path.join(HNSW_CWD, "chunks.json")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")
TOP_K = 5

# One index on disk, so uploads and searches must not overlap
index_lock = threading.Lock()


class BackendError(Exception):
    pass


# ---------- helpers ----------

def embed(texts: list[str], kind: str) -> list[list[float]]:
    """Embed texts with nomic-embed-text. kind is "document" or "query":
    the model expects a task prefix on each input."""
    prefix = "search_document: " if kind == "document" else "search_query: "
    vectors = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        try:
            r = requests.post(f"{OLLAMA_URL}/api/embed",
                              json={"model": EMBED_MODEL, "input": batch}, timeout=300)
        except requests.RequestException as e:
            raise BackendError(f"Can't reach Ollama at {OLLAMA_URL}: {e}")
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if r.status_code != 200 or "embeddings" not in data:
            raise BackendError(f"Embedding failed: {data.get('error', r.text[:200])}")
        vectors.extend(data["embeddings"])
    return vectors


def vec_to_str(vec: list[float]) -> str:
    """Serialize vector to comma-separated string for C++ subprocess."""
    return ",".join(f"{v:.6f}" for v in vec)


def run_hnsw(args: list[str], stdin: str | None = None) -> str:
    if not os.path.exists(HNSW_EXE):
        raise BackendError(f"HNSW binary not found at {HNSW_EXE}. Compile hnsw/hnsw.cpp first.")
    result = subprocess.run([HNSW_EXE, *args], cwd=HNSW_CWD, input=stdin,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise BackendError(f"hnsw {args[0]} failed: {result.stderr.strip() or result.returncode}")
    return result.stdout


def hnsw_query(cmd: str, vec: list[float], k: int) -> dict:
    """Run "search" (HNSW) or "exact" (brute force). Returns hits as
    (chunk id, distance) plus the C++ side's own timing and work counters."""
    out = run_hnsw([cmd, vec_to_str(vec), str(k)])
    stats, hits = {}, []
    for line in out.strip().split("\n"):
        parts = line.split("\t")
        if parts[0] == "#":
            stats = {"micros": float(parts[1]), "distance_calls": int(parts[2]),
                     "index_size": int(parts[3])}
        elif len(parts) == 2:
            hits.append({"id": int(parts[1]), "distance": float(parts[0])})
    return {**stats, "hits": hits}


def extract_pages(pdf) -> list[tuple[int, list[str]]]:
    """(page number, words) for each page."""
    return [(n + 1, page.get_text().split()) for n, page in enumerate(pdf)]


def chunk_pages(pages, size=300, overlap=50) -> list[dict]:
    """Split text into overlapping chunks of ~300 words, remembering pages."""
    words = [(w, n) for n, ws in pages for w in ws]
    chunks = []
    i = 0
    while i < len(words):
        window = words[i:i + size]
        text = " ".join(w for w, _ in window)
        if len(text.strip()) >= 20:
            chunks.append({"id": len(chunks), "text": text,
                           "page_start": window[0][1], "page_end": window[-1][1]})
        if i + size >= len(words):
            break  # last window reached the end; another would be a subset of it
        i += size - overlap
    return chunks


def load_doc() -> dict:
    if not os.path.exists(CHUNKS_FILE):
        return {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def page_label(c: dict) -> str:
    if c["page_start"] == c["page_end"]:
        return f"p. {c['page_start']}"
    return f"pp. {c['page_start']}-{c['page_end']}"


def stream_llm(prompt: str):
    """Yield the LLM's answer as it is generated."""
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": True
        }, stream=True, timeout=300)
    except requests.RequestException as e:
        yield f"[error] Can't reach Ollama at {OLLAMA_URL}: {e}"
        return
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            if "error" in data:
                yield f"[error] {data['error']}"
            elif "response" in data:
                yield data["response"]


@app.errorhandler(BackendError)
def backend_error(e):
    return jsonify({"error": str(e)}), 500


# ---------- routes ----------

@app.route("/upload", methods=["POST"])
def upload():
    """Upload a PDF, extract text, chunk it, embed each chunk, build the HNSW index."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
    except Exception:
        return jsonify({"error": "not a readable PDF"}), 400

    pages = extract_pages(pdf)
    chunks = chunk_pages(pages)
    if not chunks:
        return jsonify({"error": "no text found (scanned PDFs aren't supported)"}), 400

    t0 = time.perf_counter()
    vectors = embed([c["text"] for c in chunks], "document")
    t1 = time.perf_counter()

    # The index stores only chunk ids; text and pages live in chunks.json.
    # This keeps non-ASCII text out of command-line arguments.
    with index_lock:
        run_hnsw(["reset"])
        lines = "".join(f"{c['id']}\t{vec_to_str(v)}\n" for c, v in zip(chunks, vectors))
        run_hnsw(["build"], stdin=lines)
        doc = {"filename": file.filename, "pages": len(pages), "chunks": chunks}
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(doc, f)
    t2 = time.perf_counter()

    return jsonify({"status": "ok", "filename": file.filename, "pages": len(pages),
                    "chunks": len(chunks), "embed_seconds": round(t1 - t0, 2),
                    "index_seconds": round(t2 - t1, 2)})


@app.route("/status", methods=["GET"])
def status():
    """Which document is indexed right now, so the UI survives a refresh."""
    doc = load_doc()
    if not doc:
        return jsonify({"indexed": False})
    return jsonify({"indexed": True, "filename": doc["filename"],
                    "pages": doc["pages"], "chunks": len(doc["chunks"])})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Find the chunks closest to the question with HNSW and with brute force,
    so the UI can show what was retrieved and what HNSW saved."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    k = int(data.get("k", TOP_K))
    if not question:
        return jsonify({"error": "no question"}), 400

    doc = load_doc()
    if not doc:
        return jsonify({"error": "upload a PDF first"}), 400
    chunks = doc["chunks"]

    t0 = time.perf_counter()
    q_vec = embed([question], "query")[0]
    embed_ms = (time.perf_counter() - t0) * 1000

    with index_lock:
        hnsw = hnsw_query("search", q_vec, k)
        exact = hnsw_query("exact", q_vec, k)

    def describe(hits):
        return [{**h, "text": chunks[h["id"]]["text"], "pages": page_label(chunks[h["id"]])}
                for h in hits]

    exact_ids = {h["id"] for h in exact["hits"]}
    found = sum(1 for h in hnsw["hits"] if h["id"] in exact_ids)

    return jsonify({
        "embed_ms": round(embed_ms, 1),
        "hnsw": {**hnsw, "hits": describe(hnsw["hits"])},
        "exact": {**exact, "hits": describe(exact["hits"])},
        "recall": found / len(exact_ids) if exact_ids else 1.0,
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Stream an answer from llama3.2.
    mode "rag": answer from the given chunk ids (as returned by /retrieve).
    mode "plain": the same model with no document, as the baseline."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    mode = data.get("mode", "rag")
    if not question:
        return jsonify({"error": "no question"}), 400

    if mode == "plain":
        prompt = f"""You are a helpful assistant. Answer the question.

Question: {question}
Answer:"""
    else:
        chunks = load_doc().get("chunks", [])
        ids = data.get("chunk_ids")
        if ids is None:  # no retrieval done by the caller, do it here
            ids = [h["id"] for h in hnsw_query("search", embed([question], "query")[0], TOP_K)["hits"]]
        sources = [chunks[i] for i in ids if 0 <= i < len(chunks)]
        context = "\n\n".join(f"[{n}] ({page_label(c)}) {c['text']}"
                              for n, c in enumerate(sources, 1))
        prompt = f"""You are a helpful assistant. Answer the question using only the numbered sources below.
Cite the sources you use like [1] or [2].
If the answer is not in the sources, say "I don't know based on this document."

Sources:
{context}

Question: {question}
Answer:"""

    return Response(stream_with_context(stream_llm(prompt)), mimetype="text/plain")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
