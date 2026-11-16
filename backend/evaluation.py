"""Evaluation harness: measures what each part of the RAG pipeline is worth.

1. Build a test set from the indexed PDF: the LLM writes a question, a reference
   answer and a word-for-word evidence span for sampled chunks. Every chunk that
   contains the evidence span counts as relevant (chunks overlap by 50 words).
2. Answer every question with every configuration: no RAG, basic RAG (HNSW),
   exact brute-force search, and all 8 on/off combinations of the advanced
   steps (query rewriting, hybrid search, reranking).
3. Score retrieval (hit@k, recall@5, MRR, nDCG@5), answers (LLM-as-judge
   correctness, relevance and faithfulness, token F1, refusals, citations) and
   latency per stage.
4. Summarise with bootstrap 95% confidence intervals, paired differences against
   basic RAG, and the main effect of each advanced step across the factorial grid.
"""
import os
import re
import json
import math
import time
import random
import hashlib
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import agent
import pipeline
from pipeline import (BackendError, TOP_K, CANDIDATES, index_lock, load_doc, page_label,
                      embed_query, hnsw_query, advanced_retrieve, plain_prompt, rag_prompt,
                      llm_call)
from retrieval import tokenize

EVAL_DIR = os.path.join(pipeline.HNSW_CWD, "eval")
DATASET_FILE = os.path.join(EVAL_DIR, "dataset.json")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "") or None  # default: the answering model

# ---------- configurations ----------

ADVANCED_STEPS = ("rewrite", "hybrid", "rerank")
STEP_NAMES = {"rewrite": "Rewrite", "hybrid": "Hybrid", "rerank": "Rerank"}


def _advanced_configs():
    out = []
    for mask in range(1, 8):
        flags = {s: bool(mask >> i & 1) for i, s in enumerate(ADVANCED_STEPS)}
        on = [s for s in ADVANCED_STEPS if flags[s]]
        cid = "advanced" if len(on) == 3 else "+".join(on)
        name = "Advanced (all three)" if len(on) == 3 else " + ".join(STEP_NAMES[s] for s in on)
        out.append({"id": cid, "name": name, "kind": "advanced", "flags": flags})
    # Pipeline order: rewrite, hybrid, rerank, then pairs, then all three
    out.sort(key=lambda c: (sum(c["flags"].values()), [not c["flags"][s] for s in ADVANCED_STEPS]))
    return out


CONFIGS = [
    {"id": "no_rag", "name": "No RAG", "kind": "plain",
     "flags": {s: False for s in ADVANCED_STEPS}},
    {"id": "basic", "name": "Basic RAG (HNSW)", "kind": "hnsw",
     "flags": {s: False for s in ADVANCED_STEPS}},
    {"id": "exact", "name": "Exact search", "kind": "exact",
     "flags": {s: False for s in ADVANCED_STEPS}},
    *_advanced_configs(),
    {"id": "agent", "name": "Agentic RAG", "kind": "agent",
     "flags": {s: False for s in ADVANCED_STEPS}},
]
CONFIG_BY_ID = {c["id"]: c for c in CONFIGS}
REFERENCE = "basic"  # paired differences are reported against this configuration

METRICS = [
    # key, label, group, higher is better
    ("hit1", "Hit@1", "retrieval", True),
    ("hit3", "Hit@3", "retrieval", True),
    ("hit5", "Hit@5", "retrieval", True),
    ("recall5", "Recall@5", "retrieval", True),
    ("mrr", "MRR@5", "retrieval", True),
    ("ndcg5", "nDCG@5", "retrieval", True),
    ("pool_hit", "Hit@15 (candidates)", "retrieval", True),
    ("correctness", "Correctness", "answer", True),
    ("faithfulness", "Faithfulness", "answer", True),
    ("relevance", "Relevance", "answer", True),
    ("f1", "Token F1", "answer", True),
    ("refusal", "Refusal rate", "answer", False),
    ("cited", "Cites a source", "answer", True),
    ("citation_valid", "Valid citations", "answer", True),
    ("grounded", "Passed self-check", "answer", True),
    ("total_ms", "Latency (ms)", "latency", False),
]
METRIC_KEYS = [m[0] for m in METRICS]
STAGES = ("embed_ms", "plan_ms", "rewrite_ms", "search_ms", "rerank_ms", "generate_ms", "check_ms")


# ---------- text helpers ----------

def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def doc_fingerprint(doc: dict) -> str:
    h = hashlib.sha1(doc.get("filename", "").encode())
    for c in doc.get("chunks", []):
        h.update(c["text"].encode("utf-8", "ignore"))
    return h.hexdigest()[:12]


def parse_json(text: str) -> dict:
    """LLM JSON output, tolerating text around the object."""
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                out = json.loads(m.group(0))
                return out if isinstance(out, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


# ---------- 1. test set generation ----------

QUESTION_STYLES = [
    ("factual", "factual question (who, what, which, where or when)"),
    ("numeric", "question whose answer is a number, amount, date or quantity"),
    ("definition", "question asking what a term, method or concept is or means"),
    ("reasoning", "question asking why or how something happens"),
]
SELF_REFERENCE = re.compile(
    r"\b(the|this|that|above|given) (passage|text|document|excerpt|section|paragraph|author|article)\b", re.I)


def qa_prompt(chunk: dict, style: str) -> str:
    return f"""You are writing a test set for a document question-answering system.
Read the passage and write ONE {style} that the passage answers.

Rules:
- The question must make sense on its own, without seeing the passage. Never write "the passage", "the text", "this document" or "the author".
- Name the specific subject (person, product, method, place) so the question is not ambiguous in a long document.
- Where natural, use different words from the passage.
- The answer must be stated in the passage. Keep it short: a phrase or one sentence.
- "evidence" is the shortest span copied word for word from the passage that contains the answer.
- If the passage has nothing for this kind of question, write a factual question instead.

Passage ({page_label(chunk)}):
\"\"\"{chunk['text']}\"\"\"

Reply as JSON: {{"question": "...", "answer": "...", "evidence": "..."}}"""


def eligible_chunk(chunk: dict) -> bool:
    """Skip chunks that are mostly numbers, symbols or reference lists."""
    words = chunk["text"].split()
    if len(words) < 40:
        return False
    alpha = sum(1 for w in words if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*[.,;:]?", w))
    return alpha / len(words) >= 0.55


def sample_chunks(chunks: list[dict], n: int, seed: int) -> list[int]:
    """Spread the sample over the whole document: split eligible chunks into n
    equal bins and take one from each, then the rest in random order as spares
    for rejected questions."""
    rng = random.Random(seed)
    ids = [c["id"] for c in chunks if eligible_chunk(c)] or [c["id"] for c in chunks]
    if n >= len(ids):
        rng.shuffle(ids)
        return ids
    picked = []
    for b in range(n):
        lo, hi = len(ids) * b // n, len(ids) * (b + 1) // n
        picked.append(ids[rng.randrange(lo, hi)])
    spares = [i for i in ids if i not in set(picked)]
    rng.shuffle(spares)
    return picked + spares


def validate_qa(qa: dict, chunk: dict, chunks: list[dict]) -> tuple[dict | None, str]:
    """Check one generated item; return (item, "") or (None, reason it was rejected)."""
    q = str(qa.get("question", "")).strip()
    a = str(qa.get("answer", "")).strip()
    ev = str(qa.get("evidence", "")).strip()
    if not q or not a:
        return None, "missing question or answer"
    if SELF_REFERENCE.search(q):
        return None, "question refers to 'the passage'"
    if not 4 <= len(q.split()) <= 40:
        return None, "question too short or long"
    ev_norm = normalize(ev)
    chunk_norm = normalize(chunk["text"])
    if ev_norm and len(ev_norm.split()) >= 2 and ev_norm in chunk_norm:
        exact = True
    else:
        # Small models paraphrase the "verbatim" span; accept it when nearly all its
        # words occur in the chunk, but then only the source chunk counts as relevant
        ev_tokens = tokenize(ev) or tokenize(a)
        chunk_tokens = set(tokenize(chunk["text"]))
        if len(ev_tokens) < 2 or sum(t in chunk_tokens for t in ev_tokens) / len(ev_tokens) < 0.8:
            return None, "evidence not found in passage"
        exact = False
    # A short span like "in 2019" can occur anywhere, so only a span of 5+ words
    # marks other (overlapping) chunks as relevant too
    if exact and len(ev_norm.split()) >= 5:
        relevant = sorted({c["id"] for c in chunks if ev_norm in normalize(c["text"])} | {chunk["id"]})
    else:
        relevant = [chunk["id"]]
    return {"question": q if q.endswith("?") else q + "?", "answer": a, "evidence": ev,
            "source_chunk": chunk["id"], "relevant": relevant, "pages": page_label(chunk),
            "exact_evidence": exact}, ""


def too_similar(q: str, existing: list[str], threshold=0.6) -> bool:
    a = set(tokenize(q))
    for other in existing:
        b = set(tokenize(other))
        if a and b and len(a & b) / len(a | b) >= threshold:
            return True
    return False


def generate_dataset(n=25, seed=7, workers=1, progress=None, cancel=None) -> dict:
    doc = load_doc()
    if not doc:
        raise BackendError("upload a PDF first")
    chunks = doc["chunks"]
    order = sample_chunks(chunks, n, seed)
    items, rejected, attempted = [], Counter(), 0
    lock = threading.Lock()

    def attempt(pos: int, cid: int):
        style_id, style = QUESTION_STYLES[pos % len(QUESTION_STYLES)]
        out, _ = llm_call(qa_prompt(chunks[cid], style), as_json=True)
        item, reason = validate_qa(parse_json(out), chunks[cid], chunks)
        if item:
            item["style"] = style_id
        return item, reason

    queue = list(enumerate(order))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        while queue and len(items) < n:
            if cancel and cancel.is_set():
                break
            batch = queue[:max(1, workers) * 2]
            queue = queue[len(batch):]
            for item, reason in pool.map(lambda p: attempt(*p), batch):
                attempted += 1
                with lock:
                    if item and len(items) < n:
                        if too_similar(item["question"], [i["question"] for i in items]):
                            rejected["duplicate question"] += 1
                        else:
                            items.append(item)
                    elif not item:
                        rejected[reason] += 1
                if progress:
                    progress(len(items), n, f"{len(items)} of {n} questions written "
                                            f"({attempted} chunks tried)")
    items.sort(key=lambda i: i["source_chunk"])
    for k, item in enumerate(items):
        item["id"] = k
    dataset = {
        "doc": {"filename": doc["filename"], "chunks": len(chunks),
                "fingerprint": doc_fingerprint(doc)},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": pipeline.LLM_MODEL, "seed": seed,
        "stats": {"attempted": attempted, "accepted": len(items), "rejected": dict(rejected)},
        "items": items,
    }
    save_dataset(dataset)
    return dataset


def save_dataset(dataset: dict):
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=1)


def load_dataset() -> dict:
    if not os.path.exists(DATASET_FILE):
        return {}
    with open(DATASET_FILE, encoding="utf-8") as f:
        return json.load(f)


# ---------- 2. metrics ----------

def retrieval_metrics(ranked: list[int], relevant: set[int], pool: list[int] | None = None) -> dict:
    first = next((r for r, i in enumerate(ranked[:TOP_K], 1) if i in relevant), None)
    dcg = sum(1 / math.log2(r + 1) for r, i in enumerate(ranked[:TOP_K], 1) if i in relevant)
    idcg = sum(1 / math.log2(r + 1) for r in range(1, min(len(relevant), TOP_K) + 1))
    out = {
        "rank": first,
        "hit1": float(first is not None and first <= 1),
        "hit3": float(first is not None and first <= 3),
        "hit5": float(first is not None),
        "recall5": len(relevant & set(ranked[:TOP_K])) / len(relevant) if relevant else 0.0,
        "mrr": 1 / first if first else 0.0,
        "ndcg5": dcg / idcg if idcg else 0.0,
    }
    if pool is not None:
        out["pool_hit"] = float(bool(relevant & set(pool)))
    return out


REFUSAL = re.compile(
    r"i don'?t know|not (mentioned|provided|stated|specified|given|available|included)|"
    r"(do not|don'?t|does not|doesn'?t) (contain|mention|provide|say|specify)|"
    r"no (information|mention)|cannot (answer|find|determine)|unable to (answer|find|determine)", re.I)


def token_f1(answer: str, reference: str) -> float:
    a, r = Counter(tokenize(answer)), Counter(tokenize(reference))
    common = sum((a & r).values())
    if not common:
        return 0.0
    p, rec = common / sum(a.values()), common / sum(r.values())
    return 2 * p * rec / (p + rec)


def answer_metrics(answer: str, reference: str, n_sources: int | None) -> dict:
    out = {"f1": token_f1(answer, reference), "refusal": float(bool(REFUSAL.search(answer)))}
    if n_sources is not None:
        cites = [int(c) for c in re.findall(r"\[(\d+)\]", answer)]
        out["cited"] = float(bool(cites))
        out["citation_valid"] = float(all(1 <= c <= n_sources for c in cites)) if cites else None
    return out


def judge_score(data: dict, key: str) -> float | None:
    """1-5 judge score scaled to 0-1, or None if the judge didn't give one."""
    v = data.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return (min(5.0, max(1.0, v)) - 1) / 4


def judge_correctness(question: str, reference: str, answer: str) -> dict:
    out, ms = llm_call(f"""You are grading an answer from a question-answering system against a reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score each from 1 to 5:
- correctness: 5 = states the same facts as the reference (wording may differ), 3 = partly correct or incomplete, 1 = wrong, or says it doesn't know although the reference has an answer.
- relevance: 5 = directly answers the question, 3 = answers with a lot of unrelated text, 1 = does not address the question.
Reply as JSON: {{"correctness": n, "relevance": n, "reason": "one short sentence"}}""",
                       as_json=True, model=JUDGE_MODEL)
    data = parse_json(out)
    return {"correctness": judge_score(data, "correctness"),
            "relevance": judge_score(data, "relevance"),
            "reason": str(data.get("reason", ""))[:300], "ms": ms}


def judge_faithfulness(sources: list[dict], answer: str) -> dict:
    context = "\n\n".join(f"[{n}] {c['text']}" for n, c in enumerate(sources, 1))
    out, ms = llm_call(f"""You are checking whether an answer is supported by the sources it was given.

Sources:
{context}

Answer: {answer}

Score faithfulness from 1 to 5: 5 = every claim in the answer is stated in the sources, 3 = some claims are not in the sources, 1 = mostly unsupported or contradicts the sources. An answer that only says it doesn't know scores 5.
Reply as JSON: {{"faithfulness": n, "reason": "one short sentence"}}""",
                       as_json=True, model=JUDGE_MODEL)
    data = parse_json(out)
    return {"faithfulness": judge_score(data, "faithfulness"),
            "reason": str(data.get("reason", ""))[:300], "ms": ms}


# ---------- 3. running one question through one configuration ----------

def run_config(item: dict, config: dict, chunks: list[dict], q_vec, embed_ms: float,
               judge: bool) -> dict:
    question = item["question"]
    relevant = set(item["relevant"])
    lat = {s: 0.0 for s in STAGES}
    rec = {"config": config["id"], "item": item["id"]}

    answer = None
    if config["kind"] == "plain":
        sources = None
        prompt = plain_prompt(question)
    elif config["kind"] == "agent":
        res = agent.run(question, chunks)
        ranked = res["sources"]
        for s, ms in res["timings"].items():
            if s in lat:
                lat[s] = float(ms)
        rec["route"] = res["route"]
        rec["steps"] = [{k: s[k] for k in ("type", "tool", "input", "found") if k in s}
                        for s in res["steps"] if s["type"] in ("tool", "correct", "escalate")]
        rec["retrieved"] = ranked
        rec.update(retrieval_metrics(ranked, relevant))
        sources = [chunks[i] for i in ranked]
        answer = res["answer"]
        if res["check"] is not None:
            rec["grounded"] = 1.0 if res["check"]["grounded"] else 0.0
    else:
        lat["embed_ms"] = embed_ms
        if config["kind"] in ("hnsw", "exact"):
            t = time.perf_counter()
            with index_lock:
                hits = hnsw_query("search" if config["kind"] == "hnsw" else "exact", q_vec, TOP_K)["hits"]
            lat["search_ms"] = (time.perf_counter() - t) * 1000
            ranked, pool = [h["id"] for h in hits], None
        else:
            f = config["flags"]
            adv = advanced_retrieve(question, q_vec, chunks, f["rewrite"], f["hybrid"], f["rerank"], TOP_K)
            ranked, pool = adv["final"], adv["candidates"]
            for s in ("rewrite_ms", "search_ms", "rerank_ms"):
                lat[s] = float(adv["timings"].get(s, 0))
            if adv["query"] != question:
                rec["query"] = adv["query"]
        rec["retrieved"] = ranked
        if pool is not None and len(pool) > len(ranked):
            rec["candidates"] = pool
        rec.update(retrieval_metrics(ranked, relevant,
                                     pool if pool is not None and len(pool) > TOP_K else None))
        sources = [chunks[i] for i in ranked]
        prompt = rag_prompt(question, sources)

    if answer is None:
        answer, gen_ms = llm_call(prompt)
        lat["generate_ms"] = gen_ms
    rec["answer"] = answer
    rec.update(answer_metrics(answer, item["answer"], len(sources) if sources is not None else None))

    if judge:
        j = judge_correctness(question, item["answer"], answer)
        rec["correctness"], rec["relevance"] = j["correctness"], j["relevance"]
        rec["judge_reason"] = j["reason"]
        if sources is not None:
            jf = judge_faithfulness(sources, answer)
            rec["faithfulness"] = jf["faithfulness"]
            rec["faithfulness_reason"] = jf["reason"]

    lat["total_ms"] = sum(lat[s] for s in STAGES)
    rec["latency"] = {k: round(v, 1) for k, v in lat.items()}
    rec["total_ms"] = rec["latency"]["total_ms"]
    return rec


# ---------- 4. statistics ----------

def _resamples(n: int, b: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(b)]


def bootstrap(values: list[float], resamples: list[list[int]] | None = None,
              b=1000, seed=0) -> dict:
    """Mean with a 95% percentile bootstrap interval, and a two-sided bootstrap
    p-value for "the mean is 0" (used for paired differences)."""
    n = len(values)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "lo": mean, "hi": mean, "n": 1, "p": 1.0}
    resamples = resamples or _resamples(n, b, seed)
    means = sorted(sum(values[i] for i in idx) / n for idx in resamples)
    lo, hi = means[int(0.025 * len(means))], means[min(len(means) - 1, int(0.975 * len(means)))]
    below = sum(m <= 0 for m in means) / len(means)
    above = sum(m >= 0 for m in means) / len(means)
    p = 1.0 if lo == hi == 0 else min(1.0, 2 * min(below, above))
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "p": p}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (pos - lo)


def summarize(records: list[dict], items: list[dict], config_ids: list[str],
              b=1000, seed=0) -> dict:
    """Per-configuration means with CIs, paired differences against basic RAG,
    main effects of each advanced step, and latency percentiles."""
    by = {(r["config"], r["item"]): r for r in records}
    item_ids = [i["id"] for i in items if any((c, i["id"]) in by for c in config_ids)]
    cache: dict[int, list[list[int]]] = {}

    def boot(vals):
        n = len(vals)
        if n not in cache:
            cache[n] = _resamples(n, b, seed + n)
        return bootstrap(vals, cache[n])

    def values(cid, key):
        return [(i, by[(cid, i)][key]) for i in item_ids
                if (cid, i) in by and by[(cid, i)].get(key) is not None]

    configs = {}
    for cid in config_ids:
        m = {}
        for key in METRIC_KEYS:
            vals = [v for _, v in values(cid, key)]
            if vals:
                m[key] = boot(vals)
        lats = [by[(cid, i)]["latency"] for i in item_ids if (cid, i) in by]
        m["latency"] = {
            "p50": percentile([l["total_ms"] for l in lats], 0.5),
            "p95": percentile([l["total_ms"] for l in lats], 0.95),
            "stages": {s: sum(l[s] for l in lats) / len(lats) for s in STAGES} if lats else {},
        }
        configs[cid] = m

    # Paired differences: same question, this config minus basic RAG
    deltas = {}
    if REFERENCE in config_ids:
        for cid in config_ids:
            if cid == REFERENCE:
                continue
            d = {}
            for key in METRIC_KEYS:
                ref = dict(values(REFERENCE, key))
                pairs = [v - ref[i] for i, v in values(cid, key) if i in ref]
                if pairs:
                    d[key] = boot(pairs)
            deltas[cid] = d

    # Main effect of each step: average over every pair of configurations that
    # differ only in that step (up to 4 pairs when the full grid ran), per question
    adv = [c for c in config_ids if CONFIG_BY_ID[c]["kind"] in ("advanced", "hnsw")]
    effects = {}
    for step in ADVANCED_STEPS:
        pairs = []
        for on in adv:
            fo = CONFIG_BY_ID[on]["flags"]
            if not fo[step]:
                continue
            off_flags = {**fo, step: False}
            off = next((c for c in adv if CONFIG_BY_ID[c]["flags"] == off_flags), None)
            if off:
                pairs.append((on, off))
        if not pairs:
            continue
        e = {"pairs": [list(p) for p in pairs]}
        for key in METRIC_KEYS:
            per_item = []
            for i in item_ids:
                diffs = [by[(on, i)][key] - by[(off, i)][key] for on, off in pairs
                         if (on, i) in by and (off, i) in by
                         and by[(on, i)].get(key) is not None and by[(off, i)].get(key) is not None]
                if diffs:
                    per_item.append(sum(diffs) / len(diffs))
            if per_item:
                e[key] = boot(per_item)
        effects[step] = e

    # Questions no configuration could retrieve evidence for
    misses = [i for i in item_ids
              if all(not by[(c, i)].get("hit5") for c in config_ids
                     if (c, i) in by and "hit5" in by[(c, i)])
              and any("hit5" in by[(c, i)] for c in config_ids if (c, i) in by)]
    return {"configs": configs, "deltas": deltas, "effects": effects,
            "never_retrieved": misses, "questions": len(item_ids)}


# ---------- 5. full run ----------

def run_evaluation(config_ids: list[str] | None = None, judge=True, limit: int | None = None,
                   workers=1, progress=None, cancel=None, b=1000) -> dict:
    doc = load_doc()
    dataset = load_dataset()
    if not doc:
        raise BackendError("upload a PDF first")
    if not dataset.get("items"):
        raise BackendError("generate a test set first")
    if dataset["doc"]["fingerprint"] != doc_fingerprint(doc):
        raise BackendError("the test set was made for a different document; generate a new one")
    config_ids = [c["id"] for c in CONFIGS if c["id"] in (config_ids or CONFIG_BY_ID)]
    if not config_ids:
        raise BackendError("pick at least one configuration")
    items = dataset["items"][:limit] if limit else dataset["items"]
    chunks = doc["chunks"]

    total = len(items) * len(config_ids)
    done = 0
    lock = threading.Lock()
    records = []
    before = {"llm": pipeline.llm_cache.stats(), "embed": pipeline.embed_cache.stats()}
    t0 = time.perf_counter()

    def one_item(item):
        nonlocal done
        if cancel and cancel.is_set():
            return []
        q_vec, embed_ms = embed_query(item["question"])
        out = []
        for cid in config_ids:
            if cancel and cancel.is_set():
                break
            out.append(run_config(item, CONFIG_BY_ID[cid], chunks, q_vec, embed_ms, judge))
            with lock:
                done += 1
                if progress:
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / done * (total - done)
                    progress(done, total, f"Question {item['id'] + 1}: {CONFIG_BY_ID[cid]['name']} "
                                          f"· about {fmt_duration(eta)} left")
        return out

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for recs in pool.map(one_item, items):
            records.extend(recs)

    wall = time.perf_counter() - t0
    after = {"llm": pipeline.llm_cache.stats(), "embed": pipeline.embed_cache.stats()}
    complete = {i["id"] for i in items
                if all(any(r["item"] == i["id"] and r["config"] == c for r in records)
                       for c in config_ids)}
    run = {
        "id": time.strftime("%Y%m%d-%H%M%S"),
        "status": "cancelled" if cancel and cancel.is_set() else "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "doc": dataset["doc"],
        "models": {"llm": pipeline.LLM_MODEL, "embed": pipeline.EMBED_MODEL,
                   "judge": (JUDGE_MODEL or pipeline.LLM_MODEL) if judge else None},
        "settings": {"top_k": TOP_K, "candidates": CANDIDATES, "judge": judge,
                     "workers": workers, "bootstrap": b, "limit": limit},
        "configs": [CONFIG_BY_ID[c] for c in config_ids],
        "items": [i for i in items if i["id"] in complete],
        "records": [r for r in records if r["item"] in complete],
        "wall_seconds": round(wall, 1),
        "llm_calls": after["llm"]["misses"] - before["llm"]["misses"],
        "cached_calls": after["llm"]["hits"] - before["llm"]["hits"],
    }
    run["summary"] = summarize(run["records"], run["items"], config_ids, b=b)
    save_run(run)
    return run


def fmt_duration(s: float) -> str:
    if s < 90:
        return f"{max(1, round(s))} s"
    if s < 5400:
        return f"{round(s / 60)} min"
    return f"{s / 3600:.1f} h"


def save_run(run: dict):
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, f"{run['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(run, f)


def list_runs() -> list[dict]:
    if not os.path.isdir(RUNS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(RUNS_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
                run = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"id": run["id"], "created_at": run["created_at"], "status": run["status"],
                    "doc": run["doc"]["filename"], "questions": len(run["items"]),
                    "configs": len(run["configs"]), "judge": run["settings"]["judge"],
                    "wall_seconds": run["wall_seconds"]})
    return out


def load_run(run_id: str) -> dict:
    if not re.fullmatch(r"[0-9-]+", run_id or ""):
        return {}
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------- 6. markdown report ----------

def _fmt(stat: dict | None, key: str) -> str:
    if not stat or stat.get("mean") is None:
        return "–"
    if key == "total_ms":
        return f"{stat['mean']:.0f}"
    return f"{stat['mean']:.2f}"


def _fmt_delta(stat: dict | None, key: str) -> str:
    if not stat or stat.get("mean") is None:
        return "–"
    star = "*" if stat.get("p", 1) < 0.05 else ""
    if key == "total_ms":
        return f"{stat['mean']:+.0f}{star}"
    return f"{stat['mean']:+.2f} [{stat['lo']:+.2f}, {stat['hi']:+.2f}]{star}"


def markdown_report(run: dict) -> str:
    s = run["summary"]
    cids = [c["id"] for c in run["configs"]]
    names = {c["id"]: c["name"] for c in run["configs"]}
    judge = run["settings"]["judge"]
    keys = ["hit1", "hit5", "mrr", "ndcg5"] + (["correctness", "faithfulness", "relevance"] if judge else []) \
        + ["f1", "refusal", "total_ms"]
    label = {k: lbl for k, lbl, _, _ in METRICS}
    lines = [
        f"# RAG evaluation · {run['doc']['filename']}",
        "",
        f"{len(run['items'])} questions × {len(cids)} configurations, run {run['created_at']} "
        f"({run['status']}, {fmt_duration(run['wall_seconds'])}, {run['llm_calls']} LLM calls, "
        f"{run['cached_calls']} reused from cache).",
        f"Answer model `{run['models']['llm']}`, embeddings `{run['models']['embed']}`"
        + (f", judge `{run['models']['judge']}`." if judge else ", no LLM judge."),
        "",
        "## Scores",
        "",
        "Means over questions; 95% bootstrap intervals are in the JSON. Latency is the mean end-to-end time in ms.",
        "",
        "| Configuration | " + " | ".join(label[k] for k in keys) + " | p95 ms |",
        "|---|" + "---:|" * (len(keys) + 1),
    ]
    for cid in cids:
        m = s["configs"][cid]
        p95 = m["latency"]["p95"]
        p95_text = f"{p95:.0f}" if p95 is not None else "–"
        lines.append(f"| {names[cid]} | " + " | ".join(_fmt(m.get(k), k) for k in keys)
                     + f" | {p95_text} |")
    if s["deltas"]:
        lines += ["", f"## Difference from {names.get(REFERENCE, REFERENCE)}", "",
                  "Paired over the same questions, with 95% bootstrap intervals. * means p < 0.05.", "",
                  "| Configuration | " + " | ".join(label[k] for k in keys) + " |",
                  "|---|" + "---:|" * len(keys)]
        for cid, d in s["deltas"].items():
            lines.append(f"| {names[cid]} | " + " | ".join(_fmt_delta(d.get(k), k) for k in keys) + " |")
    if s["effects"]:
        lines += ["", "## What each advanced step adds", "",
                  "Main effect: the average change from turning the step on, over every pair of "
                  "configurations that differ only in that step.", "",
                  "| Step | pairs | " + " | ".join(label[k] for k in keys) + " |",
                  "|---|---:|" + "---:|" * len(keys)]
        for step, e in s["effects"].items():
            lines.append(f"| {STEP_NAMES[step]} | {len(e['pairs'])} | "
                         + " | ".join(_fmt_delta(e.get(k), k) for k in keys) + " |")
    lines += ["", "## Where the time goes", "", "Mean ms per stage.", "",
              "| Configuration | " + " | ".join(st.replace("_ms", "") for st in STAGES) + " | p50 | p95 |",
              "|---|" + "---:|" * (len(STAGES) + 2)]
    for cid in cids:
        lat = s["configs"][cid]["latency"]
        if not lat["stages"]:
            continue
        lines.append(f"| {names[cid]} | " + " | ".join(f"{lat['stages'][st]:.0f}" for st in STAGES)
                     + f" | {lat['p50']:.0f} | {lat['p95']:.0f} |")
    if s["never_retrieved"]:
        qs = {i["id"]: i for i in run["items"]}
        lines += ["", "## Questions no configuration retrieved evidence for", ""]
        lines += [f"- {qs[i]['question']} ({qs[i]['pages']})" for i in s["never_retrieved"] if i in qs]
    lines += ["", "## How to read this", "",
              "- Questions, reference answers and evidence spans were written by the LLM from sampled chunks, "
              "so they lean towards wording found in the document, which favours keyword search.",
              "- Every chunk containing the evidence span counts as relevant. Hit@k: a relevant chunk is in the top k. "
              "MRR@5: 1 / rank of the first relevant chunk. Hit@15 (candidates): the evidence was in the pool that "
              "reranking chose from, which caps what reranking can achieve.",
              "- Judge scores are 1–5 scaled to 0–1. A model judging its own answers is lenient; set `JUDGE_MODEL` "
              "to a larger model for stricter grading.",
              "- Answers use temperature 0 so runs are repeatable; the app's streamed answers use the model default.",
              "- Results reused from cache report the latency of the original call.", ""]
    return "\n".join(lines)


# ---------- background jobs ----------

class Jobs:
    """One evaluation job at a time (dataset generation or a run), on a thread,
    with progress the UI can poll."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "idle"}
        self.cancel = threading.Event()

    def busy(self) -> bool:
        with self.lock:
            return self.state.get("status") == "running"

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.state)

    def start(self, kind: str, fn, **kwargs) -> bool:
        with self.lock:
            if self.state.get("status") == "running":
                return False
            self.cancel = threading.Event()
            self.state = {"status": "running", "kind": kind, "done": 0, "total": 0,
                          "message": "Starting…", "started": time.time()}
        cancel = self.cancel

        def progress(done, total, message):
            with self.lock:
                self.state.update(done=done, total=total, message=message)

        def target():
            try:
                result = fn(progress=progress, cancel=cancel, **kwargs)
                with self.lock:
                    self.state.update(status="cancelled" if cancel.is_set() else "done",
                                      result_id=result.get("id"), finished=time.time())
            except Exception as e:  # report any failure to the UI instead of dying silently
                with self.lock:
                    self.state.update(status="error", error=str(e), finished=time.time())

        threading.Thread(target=target, daemon=True).start()
        return True

    def stop(self):
        self.cancel.set()


jobs = Jobs()
