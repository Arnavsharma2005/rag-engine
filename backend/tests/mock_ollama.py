"""A stand-in for Ollama, for tests and for trying the evaluation harness
without downloading models. Standard library only.

    python tests/mock_ollama.py --port 11434

Embeddings are hashed bags of words, so similar text gets similar vectors.
/api/generate recognises each prompt the backend sends (test set writing,
query rewriting, reranking, answering, judging) and answers it with simple
word-overlap heuristics. The numbers it produces exercise the code; they say
nothing about how llama3.2 performs.
"""
import re
import sys
import json
import math
import time
import hashlib
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 768
STOP = set("""a an and are as at be by for from has have how i in is it its of on or
that the this to was were what when where which who why will with does do did about known""".split())
DELAY = 0.002  # seconds per call, so latency numbers aren't all zero


def words(text):
    return [w.rstrip("s") if len(w) > 3 else w
            for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def embed_one(text):
    text = re.sub(r"^search_(query|document): ", "", text)
    v = [0.0] * DIM
    for w in words(text):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 12) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 6]


def overlap(a, b):
    a, b = set(words(a)), set(words(b))
    return len(a & b) / (len(a) or 1)


def f1(a, b):
    a, b = words(a), words(b)
    common = len(set(a) & set(b))
    if not common:
        return 0.0
    p, r = common / len(set(a)), common / len(set(b))
    return 2 * p * r / (p + r)


def field(prompt, name):
    m = re.search(rf"^{name}: (.*)$", prompt, re.M)
    return m.group(1).strip() if m else ""


def generate(prompt):
    if prompt.startswith("Route a user's question"):
        q = field(prompt, "Question").lower()
        route = "summary" if "summar" in q else "research" if " and " in q or "compare" in q else "lookup"
        return json.dumps({"route": route, "reason": "keyword rules"})
    if prompt.startswith("You are a research agent"):
        q = field(prompt, "Question")
        steps = int(field(prompt, "Steps taken").split()[0] or 0)
        if steps == 0:
            return json.dumps({"thought": "nothing yet", "action": "search", "input": q})
        if steps == 1:
            return json.dumps({"thought": "exact terms", "action": "keyword", "input": q})
        return json.dumps({"thought": "enough", "action": "answer", "input": ""})
    if prompt.startswith("Check each claim"):
        passages = prompt.split("Passages:", 1)[1].split("Claims:", 1)[0]
        claims = re.findall(r"^(\d+)\. (.*)$", prompt.split("Claims:", 1)[1], re.M)
        return json.dumps({"claims": [{"n": int(n), "supported": overlap(c, passages) > 0.6}
                                      for n, c in claims]})
    if prompt.startswith("Summarize the passages"):
        text = prompt.split("Passages:", 1)[1]
        return " ".join((sentences(text) or [text.strip()])[:2])
    if prompt.startswith("Combine these partial summaries"):
        return " ".join(re.findall(r"^- (.*)$", prompt, re.M))
    if "writing a test set" in prompt:
        passage = prompt.split('"""')[1]
        sents = sentences(passage) or [passage]
        # Prefer sentences with names and numbers, like a real question writer would
        sents.sort(key=lambda t: -len(re.findall(r"\b[A-Z0-9]\w*", t[1:])))
        top = sents[:3]
        pick = top[int(hashlib.md5(passage.encode()).hexdigest(), 16) % len(top)]
        keys = sorted(set(words(pick)), key=lambda w: (-len(w), w))[:4]
        return json.dumps({"question": f"What is known about {', '.join(keys)}?",
                           "answer": pick, "evidence": pick})
    if "Rewrite the question" in prompt:
        return field(prompt, "Question")
    if "Rank the passages" in prompt:
        q = field(prompt, "Question")
        passages = re.findall(r"^\[(\d+)\] (.*)$", prompt, re.M)
        ranked = sorted(passages, key=lambda p: -overlap(q, p[1]))
        return json.dumps({"ranking": [int(i) for i, _ in ranked]})
    if "grading an answer" in prompt:
        ref, ans = field(prompt, "Reference answer"), field(prompt, "System answer")
        q = field(prompt, "Question")
        score = 1 + round(4 * f1(ans, ref))
        return json.dumps({"correctness": score, "relevance": 5 if overlap(q, ans) > 0.2 else 2,
                           "reason": "word overlap with the reference"})
    if "supported by the sources" in prompt:
        ans = prompt.rsplit("Answer:", 1)[1].split("Score faithfulness")[0]
        if "don't know" in ans:
            return json.dumps({"faithfulness": 5, "reason": "declined"})
        sources = prompt.split("Sources:", 1)[1].rsplit("Answer:", 1)[0]
        return json.dumps({"faithfulness": 1 + round(4 * overlap(ans, sources)),
                           "reason": "share of answer words found in the sources"})
    if "numbered sources" in prompt:
        q = prompt.rsplit("Question:", 1)[1].split("\nAnswer:")[0].strip()
        best, cite, score = None, 0, 0.0
        for n, text in re.findall(r"^\[(\d+)\] \([^)]*\) (.*)$", prompt, re.M):
            for s in sentences(text):
                o = overlap(q, s)
                if o > score:
                    best, cite, score = s, n, o
        if not best or score < 0.5:
            return "I don't know based on this document."
        return f"{best} [{cite}]"
    return "I don't know."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        time.sleep(DELAY)
        if self.path == "/api/embed":
            inputs = data["input"] if isinstance(data["input"], list) else [data["input"]]
            return self.reply({"embeddings": [embed_one(t) for t in inputs]})
        if self.path == "/api/generate":
            text = generate(data.get("prompt", ""))
            if data.get("stream", True):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                for tok in re.findall(r"\S+\s*", text):
                    self.wfile.write(json.dumps({"response": tok}).encode() + b"\n")
                self.wfile.write(json.dumps({"done": True}).encode() + b"\n")
                return
            return self.reply({"response": text, "done": True})
        self.reply({"error": "not found"}, 404)


def serve(port=0):
    """Start in a background thread; returns the server (server.server_port is the port)."""
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11434)
    port = ap.parse_args().port
    print(f"mock Ollama on http://127.0.0.1:{port}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
