"""The RAG pipeline itself: embedding, HNSW search, LLM calls, and the advanced
retrieval steps. Shared by the Flask API (app.py) and the evaluation harness
(evaluation.py), so what gets evaluated is exactly what the app runs."""
import os
import json
import time
import threading
import subprocess
from collections import OrderedDict
import requests
from retrieval import bm25_search, rrf_fuse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HNSW_CWD = os.environ.get("HNSW_DIR", os.path.join(ROOT, "hnsw"))
HNSW_EXE = os.environ.get(
    "HNSW_EXE", os.path.join(HNSW_CWD, "hnsw.exe" if os.name == "nt" else "hnsw")
)
CHUNKS_FILE = os.path.join(HNSW_CWD, "chunks.json")
# "ollama" (default, fully local) or "openai": any OpenAI-compatible API
# (OpenAI, Groq, Together, vLLM, LM Studio...) at OPENAI_BASE_URL
PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3.2")
TOP_K = 5
CANDIDATES = 15  # how many chunks the advanced pipeline gathers before reranking

# One index on disk, so uploads and searches must not overlap
index_lock = threading.Lock()


class BackendError(Exception):
    pass


# ---------- cache for deterministic calls ----------

class TimedCache:
    """LRU cache for deterministic model calls (temperature 0 generations, query
    embeddings). It stores how long the original call took, so latency numbers
    stay honest when the evaluation reuses a result: a cache hit reports the time
    the real call needed, not ~0 ms."""

    def __init__(self, size=4096):
        self.size = size
        self.data = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                self.hits += 1
                return self.data[key]
            self.misses += 1
            return None

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.data.move_to_end(key)
            while len(self.data) > self.size:
                self.data.popitem(last=False)

    def stats(self):
        with self.lock:
            return {"hits": self.hits, "misses": self.misses, "entries": len(self.data)}


llm_cache = TimedCache()
embed_cache = TimedCache()


# ---------- embeddings and HNSW ----------

def _openai_headers() -> dict:
    return {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}


def _post(url: str, body: dict, **kw):
    try:
        return requests.post(url, json=body, **kw)
    except requests.RequestException as e:
        where = OLLAMA_URL if PROVIDER == "ollama" else OPENAI_BASE_URL
        raise BackendError(f"Can't reach the model server at {where}: {e}")


def _json(r) -> dict:
    try:
        data = r.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """Embed texts. kind is "document" or "query": nomic-embed-text expects a
    task prefix on each input, other models get the text as is."""
    prefix = ""
    if "nomic" in EMBED_MODEL:
        prefix = "search_document: " if kind == "document" else "search_query: "
    vectors = []
    for i in range(0, len(texts), 32):
        batch = [prefix + t for t in texts[i:i + 32]]
        if PROVIDER == "openai":
            r = _post(f"{OPENAI_BASE_URL}/embeddings", {"model": EMBED_MODEL, "input": batch},
                      headers=_openai_headers(), timeout=300)
            data = _json(r)
            if r.status_code != 200 or "data" not in data:
                raise BackendError(f"Embedding failed: {data.get('error', r.text[:200])}")
            vectors.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
            continue
        r = _post(f"{OLLAMA_URL}/api/embed", {"model": EMBED_MODEL, "input": batch}, timeout=300)
        data = _json(r)
        if r.status_code != 200 or "embeddings" not in data:
            raise BackendError(f"Embedding failed: {data.get('error', r.text[:200])}")
        vectors.extend(data["embeddings"])
    return vectors


def embed_query(text: str) -> tuple[list[float], float]:
    """Embed one search query, cached. Returns (vector, milliseconds the embedding took)."""
    key = (EMBED_MODEL, text)
    hit = embed_cache.get(key)
    if hit:
        return hit
    t = time.perf_counter()
    vec = embed([text], "query")[0]
    result = (vec, (time.perf_counter() - t) * 1000)
    embed_cache.put(key, result)
    return result


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


# ---------- documents ----------

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


# ---------- LLM ----------

def plain_prompt(question: str) -> str:
    return f"""You are a helpful assistant. Answer the question.

Question: {question}
Answer:"""


def rag_prompt(question: str, sources: list[dict]) -> str:
    context = "\n\n".join(f"[{n}] ({page_label(c)}) {c['text']}"
                          for n, c in enumerate(sources, 1))
    return f"""You are a helpful assistant. Answer the question using only the numbered sources below.
Cite the sources you use like [1] or [2].
If the answer is not in the sources, say "I don't know based on this document."

Sources:
{context}

Question: {question}
Answer:"""


def stream_llm(prompt: str):
    """Yield the LLM's answer as it is generated."""
    try:
        if PROVIDER == "openai":
            r = _post(f"{OPENAI_BASE_URL}/chat/completions", {
                "model": LLM_MODEL, "stream": True,
                "messages": [{"role": "user", "content": prompt}]},
                headers=_openai_headers(), stream=True, timeout=300)
            if r.status_code != 200:
                yield f"[error] {_json(r).get('error', r.text[:200])}"
                return
            for line in r.iter_lines():
                line = line.decode() if isinstance(line, bytes) else line
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
            return
        r = _post(f"{OLLAMA_URL}/api/generate", {"model": LLM_MODEL, "prompt": prompt,
                                                 "stream": True}, stream=True, timeout=300)
    except BackendError as e:
        yield f"[error] {e}"
        return
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            if "error" in data:
                yield f"[error] {data['error']}"
            elif "response" in data:
                yield data["response"]


def _generate(prompt: str, as_json: bool, model: str) -> str:
    """One non-streaming, temperature 0 completion from the configured provider."""
    if PROVIDER == "openai":
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]}
        if as_json:
            body["response_format"] = {"type": "json_object"}
        r = _post(f"{OPENAI_BASE_URL}/chat/completions", body, headers=_openai_headers(), timeout=600)
        data = _json(r)
        if r.status_code != 200 or "choices" not in data:
            raise BackendError(f"LLM call failed: {data.get('error', r.text[:200])}")
        return (data["choices"][0]["message"].get("content") or "").strip()
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}}
    if as_json:
        body["format"] = "json"
    r = _post(f"{OLLAMA_URL}/api/generate", body, timeout=600)
    data = _json(r)
    if not data:
        raise BackendError(f"LLM call failed: {r.text[:200]}")
    if "error" in data:
        raise BackendError(f"LLM call failed: {data['error']}")
    return data.get("response", "").strip()


def llm_call(prompt: str, as_json=False, model: str | None = None) -> tuple[str, float]:
    """One non-streaming, temperature 0 LLM call, cached because it is deterministic.
    Returns (text, milliseconds the generation took)."""
    model = model or LLM_MODEL
    key = (model, prompt, as_json)
    hit = llm_cache.get(key)
    if hit:
        return hit
    t = time.perf_counter()
    result = (_generate(prompt, as_json, model), (time.perf_counter() - t) * 1000)
    llm_cache.put(key, result)
    return result


def llm_complete(prompt: str, as_json=False) -> str:
    """One non-streaming LLM call, for pipeline steps like rewriting and reranking."""
    return llm_call(prompt, as_json)[0]


# ---------- advanced retrieval ----------

def rewrite_query(question: str) -> tuple[str, float]:
    """Turn a conversational question into a keyword-rich search query.
    Returns (query, milliseconds)."""
    out, ms = llm_call(f"""Rewrite the question below as a short search query for finding the
relevant passage in a document. Keep names, numbers and technical terms exactly.
Add likely synonyms. Reply with the query only, on one line.

Question: {question}
Query:""")
    line = out.splitlines()[0].strip().strip('"') if out else ""
    return line or question, ms


def rerank(question: str, chunks: list[dict], ids: list[int], k: int) -> tuple[list[int], float]:
    """Ask the LLM which candidates actually answer the question. Embedding
    similarity finds related text; the LLM can judge whether it is useful.
    Returns (ids, milliseconds)."""
    listing = "\n\n".join(f"[{i}] {' '.join(chunks[i]['text'].split()[:120])}" for i in ids)
    out, ms = llm_call(f"""Below are numbered passages from a document and a question.
Rank the passages by how useful they are for answering the question, most useful first.
Reply as JSON: {{"ranking": [passage numbers]}}

Passages:
{listing}

Question: {question}""", as_json=True)
    try:
        ranking = json.loads(out).get("ranking", [])
    except (json.JSONDecodeError, AttributeError):
        ranking = []
    # Keep only valid candidate ids, then fill with the original order if the LLM skipped some
    picked = []
    for i in ranking:
        if isinstance(i, int) and i in ids and i not in picked:
            picked.append(i)
    picked += [i for i in ids if i not in picked]
    return picked[:k], ms


def advanced_retrieve(question: str, q_vec: list[float], chunks: list[dict],
                      use_rewrite: bool, use_hybrid: bool, use_rerank: bool, k: int) -> dict:
    """Run the advanced pipeline and record every stage so the UI can show what each one changed."""
    timings = {}
    query = question
    if use_rewrite:
        query, rewrite_ms = rewrite_query(question)
        q_vec, embed_ms = embed_query(query)
        timings["rewrite_ms"] = round(rewrite_ms + embed_ms)

    pool = CANDIDATES if (use_rerank or use_hybrid) else k
    t = time.perf_counter()
    with index_lock:
        vector = hnsw_query("search", q_vec, pool)["hits"]
    keyword = bm25_search(chunks, query, pool) if use_hybrid else None
    if use_hybrid:
        fused = rrf_fuse([[h["id"] for h in vector], [h["id"] for h in keyword]], pool)
    else:
        fused = [{"id": h["id"], "score": 1 - h["distance"]} for h in vector]
    timings["search_ms"] = round((time.perf_counter() - t) * 1000, 1)

    ids = [h["id"] for h in fused]
    reranked = None
    if use_rerank:
        reranked, rerank_ms = rerank(question, chunks, ids, k)
        timings["rerank_ms"] = round(rerank_ms)
        final = reranked
    else:
        final = ids[:k]

    return {"query": query, "vector": vector, "keyword": keyword,
            "fused": fused if use_hybrid else None, "reranked": reranked,
            "candidates": ids, "final": final, "timings": timings}
