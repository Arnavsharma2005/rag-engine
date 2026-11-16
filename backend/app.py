import os
import json
import time
import queue
import threading
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from pipeline import (BackendError, CHUNKS_FILE, TOP_K, index_lock, embed, embed_query,
                      vec_to_str, run_hnsw, hnsw_query, extract_pages, chunk_pages, load_doc,
                      page_label, plain_prompt, rag_prompt, stream_llm, advanced_retrieve)
import evaluation
import agent

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB uploads

# Only the local frontend may call the API, so other websites can't read your document
CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])


@app.errorhandler(BackendError)
def backend_error(e):
    return jsonify({"error": str(e)}), 500


# ---------- routes ----------

@app.route("/upload", methods=["POST"])
def upload():
    """Upload a PDF, extract text, chunk it, embed each chunk, build the HNSW index."""
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    if evaluation.jobs.busy():
        return jsonify({"error": "an evaluation is running on this document; wait or cancel it first"}), 409

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
    so the UI can show what was retrieved and what HNSW saved. If "advanced"
    is given ({rewrite, hybrid, rerank} flags), also run the advanced pipeline."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    k = int(data.get("k", TOP_K))
    if not question:
        return jsonify({"error": "no question"}), 400

    doc = load_doc()
    if not doc:
        return jsonify({"error": "upload a PDF first"}), 400
    chunks = doc["chunks"]

    q_vec, embed_ms = embed_query(question)

    with index_lock:
        hnsw = hnsw_query("search", q_vec, k)
        exact = hnsw_query("exact", q_vec, k)

    def describe(hits):
        return [{**h, "text": chunks[h["id"]]["text"], "pages": page_label(chunks[h["id"]])}
                for h in hits]

    exact_ids = {h["id"] for h in exact["hits"]}
    found = sum(1 for h in hnsw["hits"] if h["id"] in exact_ids)

    result = {
        "embed_ms": round(embed_ms, 1),
        "hnsw": {**hnsw, "hits": describe(hnsw["hits"])},
        "exact": {**exact, "hits": describe(exact["hits"])},
        "recall": found / len(exact_ids) if exact_ids else 1.0,
    }

    flags = data.get("advanced")
    if flags:
        adv = advanced_retrieve(question, q_vec, chunks, bool(flags.get("rewrite")),
                                bool(flags.get("hybrid")), bool(flags.get("rerank")), k)
        for stage in ("vector", "keyword", "fused"):
            for h in adv[stage] or []:
                h["pages"] = page_label(chunks[h["id"]])
        basic_ids = {h["id"] for h in hnsw["hits"]}
        adv["final"] = [{"id": i, "text": chunks[i]["text"], "pages": page_label(chunks[i]),
                         "new": i not in basic_ids} for i in adv["final"]]
        result["advanced"] = adv

    return jsonify(result)


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
        prompt = plain_prompt(question)
    else:
        chunks = load_doc().get("chunks", [])
        ids = data.get("chunk_ids")
        if ids is None:  # no retrieval done by the caller, do it here
            with index_lock:
                ids = [h["id"] for h in hnsw_query("search", embed_query(question)[0], TOP_K)["hits"]]
        sources = [chunks[i] for i in ids if 0 <= i < len(chunks)]
        prompt = rag_prompt(question, sources)

    return Response(stream_with_context(stream_llm(prompt)), mimetype="text/plain")


@app.route("/agent", methods=["POST"])
def agent_ask():
    """Answer with the agentic pipeline: {question, max_steps?, self_check?}.
    Streams newline-delimited JSON events (route, tool, map, escalate, correct,
    check, answer) so the UI can show each decision as it happens."""
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "no question"}), 400
    doc = load_doc()
    if not doc:
        return jsonify({"error": "upload a PDF first"}), 400
    max_steps = max(1, min(int(data.get("max_steps", agent.MAX_STEPS)), 8))
    events: queue.Queue = queue.Queue()

    def work():
        try:
            agent.run(question, doc["chunks"], doc["filename"], emit=events.put,
                      max_steps=max_steps, self_check=bool(data.get("self_check", True)))
        except Exception as e:  # report the failure to the client instead of hanging
            events.put({"type": "error", "error": str(e)})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")


# ---------- evaluation ----------

@app.route("/eval/configs", methods=["GET"])
def eval_configs():
    return jsonify({"configs": evaluation.CONFIGS, "metrics": [
        {"key": k, "label": l, "group": g, "higher_better": h} for k, l, g, h in evaluation.METRICS],
        "reference": evaluation.REFERENCE})


@app.route("/eval/dataset", methods=["GET"])
def eval_dataset():
    """The current test set, and whether it still matches the indexed document."""
    dataset = evaluation.load_dataset()
    doc = load_doc()
    current = bool(dataset and doc and
                   dataset["doc"]["fingerprint"] == evaluation.doc_fingerprint(doc))
    return jsonify({**dataset, "current": current})


@app.route("/eval/dataset", methods=["POST"])
def eval_generate():
    """Start writing a test set from the indexed PDF: {n?, seed?, workers?}."""
    if not load_doc():
        return jsonify({"error": "upload a PDF first"}), 400
    data = request.get_json(silent=True) or {}
    n = max(1, min(int(data.get("n", 25)), 200))
    started = evaluation.jobs.start("dataset", evaluation.generate_dataset, n=n,
                                    seed=int(data.get("seed", 7)),
                                    workers=max(1, min(int(data.get("workers", 1)), 8)))
    if not started:
        return jsonify({"error": "another evaluation job is running"}), 409
    return jsonify({"started": True})


@app.route("/eval/dataset/<int:item_id>", methods=["DELETE"])
def eval_delete_item(item_id):
    """Drop a bad question from the test set."""
    if evaluation.jobs.busy():
        return jsonify({"error": "wait for the running job to finish"}), 409
    dataset = evaluation.load_dataset()
    items = [i for i in dataset.get("items", []) if i["id"] != item_id]
    if len(items) == len(dataset.get("items", [])):
        return jsonify({"error": "no such question"}), 404
    dataset["items"] = items
    evaluation.save_dataset(dataset)
    return jsonify({"remaining": len(items)})


@app.route("/eval/run", methods=["POST"])
def eval_run():
    """Start an evaluation run: {configs?, judge?, limit?, workers?}."""
    data = request.get_json(silent=True) or {}
    dataset = evaluation.load_dataset()
    doc = load_doc()
    if not dataset.get("items"):
        return jsonify({"error": "generate a test set first"}), 400
    if not doc or dataset["doc"]["fingerprint"] != evaluation.doc_fingerprint(doc):
        return jsonify({"error": "the test set was made for a different document"}), 400
    limit = data.get("limit")
    started = evaluation.jobs.start(
        "run", evaluation.run_evaluation, config_ids=data.get("configs"),
        judge=bool(data.get("judge", True)), limit=int(limit) if limit else None,
        workers=max(1, min(int(data.get("workers", 1)), 8)))
    if not started:
        return jsonify({"error": "another evaluation job is running"}), 409
    return jsonify({"started": True})


@app.route("/eval/job", methods=["GET"])
def eval_job():
    return jsonify(evaluation.jobs.snapshot())


@app.route("/eval/cancel", methods=["POST"])
def eval_cancel():
    evaluation.jobs.stop()
    return jsonify({"cancelling": evaluation.jobs.busy()})


@app.route("/eval/runs", methods=["GET"])
def eval_runs():
    return jsonify({"runs": evaluation.list_runs()})


@app.route("/eval/runs/<run_id>", methods=["GET"])
def eval_run_result(run_id):
    run = evaluation.load_run(run_id)
    if not run:
        return jsonify({"error": "no such run"}), 404
    return jsonify(run)


@app.route("/eval/runs/<run_id>/report.md", methods=["GET"])
def eval_report(run_id):
    run = evaluation.load_run(run_id)
    if not run:
        return jsonify({"error": "no such run"}), 404
    return Response(evaluation.markdown_report(run), mimetype="text/markdown",
                    headers={"Content-Disposition": f"attachment; filename=rag-eval-{run_id}.md"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
