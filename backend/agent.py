"""Agentic RAG: decide how to answer each question, gather evidence with tools,
then check the answer against its sources before returning it.

    question
      -> route        direct | lookup | research | summary
      -> lookup:      one hybrid search, answer, self-check; escalates to
                      research if the answer is a refusal or fails the check
      -> research:    a tool-using loop (search, keyword, read_neighbors,
                      read_page) that runs until the agent says it has enough
                      evidence or hits MAX_STEPS
      -> summary:     map-reduce over chunks spread through the document
      -> direct:      no retrieval (greetings, general knowledge)
      -> self-check:  each cited claim is checked against its sources; if a
                      claim is unsupported, search again for it (corrective
                      RAG) and rewrite the answer once

Every step is reported through `emit`, so the UI can show the trace live and
the evaluation harness can score the agent like any other configuration.
"""
import re
import json
import time
from collections import Counter

from pipeline import (BackendError, TOP_K, index_lock, embed_query, hnsw_query, page_label,
                      plain_prompt, rag_prompt, llm_call)
from retrieval import bm25_search, rrf_fuse

MAX_STEPS = 4      # tool calls in the research loop
MAX_SOURCES = 6    # chunks handed to the answer prompt
SUMMARY_CHUNKS = 12
ROUTES = ("direct", "lookup", "research", "summary")
TOOLS = {
    "search": "hybrid vector + keyword search; input is a search query",
    "keyword": "exact keyword search (BM25), best for names, numbers and codes; input is the keywords",
    "read_neighbors": "read the chunks just before and after a chunk; input is a chunk id like 12",
    "read_page": "read every chunk on a page; input is a page number",
}
REFUSAL = re.compile(
    r"i don'?t know|not (mentioned|provided|stated|specified|given|available|included)|"
    r"(do not|don'?t|does not|doesn'?t) (contain|mention|provide|say|specify)|"
    r"no (information|mention)|cannot (answer|find|determine)|unable to (answer|find|determine)", re.I)


def parse_json(text: str) -> dict:
    """The first JSON object in an LLM reply, or {}."""
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        if m:
            try:
                data = json.loads(m.group(0))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def preview(chunk: dict, words=60) -> str:
    w = chunk["text"].split()
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


class Agent:
    def __init__(self, chunks: list[dict], filename: str = "the document", emit=None,
                 max_steps: int = MAX_STEPS, self_check: bool = True):
        self.chunks = chunks
        self.filename = filename
        self.emit = emit or (lambda event: None)
        self.max_steps = max_steps
        self.self_check = self_check
        self.scores = Counter()   # chunk id -> fused rank score over every tool call
        self.steps: list[dict] = []
        self.timings = Counter()
        self.llm_calls = 0

    # ---------- bookkeeping ----------

    def llm(self, prompt: str, as_json=False, stage="plan_ms") -> str:
        out, ms = llm_call(prompt, as_json)
        self.timings[stage] += ms
        self.llm_calls += 1
        return out

    def add_evidence(self, ids: list[int]):
        for rank, i in enumerate(ids):
            self.scores[i] += 1 / (60 + rank + 1)

    def sources(self) -> list[int]:
        return [i for i, _ in self.scores.most_common(MAX_SOURCES)]

    def step(self, kind: str, **data):
        event = {"type": kind, **data}
        self.steps.append(event)
        self.emit(event)

    # ---------- tools ----------

    def tool(self, name: str, arg) -> list[int]:
        t = time.perf_counter()
        n = len(self.chunks)
        if name == "search":
            vec, ms = embed_query(str(arg))
            self.timings["embed_ms"] += ms
            with index_lock:
                vector = [h["id"] for h in hnsw_query("search", vec, 15)["hits"]]
            keyword = [h["id"] for h in bm25_search(self.chunks, str(arg), 15)]
            ids = [h["id"] for h in rrf_fuse([vector, keyword], TOP_K)]
        elif name == "keyword":
            ids = [h["id"] for h in bm25_search(self.chunks, str(arg), TOP_K)]
        elif name == "read_neighbors":
            c = int(re.sub(r"\D", "", str(arg)) or -1)
            ids = [i for i in (c - 1, c, c + 1) if 0 <= i < n]
        elif name == "read_page":
            p = int(re.sub(r"\D", "", str(arg)) or 0)
            ids = [c["id"] for c in self.chunks if c["page_start"] <= p <= c["page_end"]][:4]
        else:
            raise ValueError(name)
        self.timings["search_ms"] += (time.perf_counter() - t) * 1000
        return ids

    # ---------- routing ----------

    def route(self, question: str) -> tuple[str, str]:
        out = self.llm(f"""Route a user's question about the document "{self.filename}".
Pick one route:
- direct: small talk or general knowledge that does not need the document
- lookup: one specific fact that a single passage should answer
- research: needs several facts, a comparison, multiple steps or reasoning across sections
- summary: asks for an overview or summary of the whole document or a large part of it
Reply as JSON: {{"route": "direct|lookup|research|summary", "reason": "a few words"}}

Question: {question}""", as_json=True)
        data = parse_json(out)
        r = str(data.get("route", "")).lower()
        if r in ROUTES:
            return r, str(data.get("reason", ""))
        # The model didn't answer in the expected shape: fall back to simple rules
        q = question.lower()
        if re.search(r"\b(summar|overview|tl;?dr|main points)", q):
            return "summary", "fallback rule"
        if re.search(r"\b(compare|difference|versus|vs\.?|both|and then|relationship)\b", q):
            return "research", "fallback rule"
        return "lookup", "fallback rule"

    # ---------- research loop ----------

    def plan(self, question: str) -> dict:
        evidence = "\n".join(f"[#{i}] ({page_label(self.chunks[i])}) {preview(self.chunks[i])}"
                             for i in self.sources()) or "(none yet)"
        history = "\n".join(f"{n}. {s['tool']}({s['input']!r}) -> chunks {s['found']}"
                            for n, s in enumerate((s for s in self.steps if s["type"] == "tool"), 1)) \
            or "(none yet)"
        tools = "\n".join(f"- {k}: {v}" for k, v in TOOLS.items())
        prompt = f"""You are a research agent answering a question from the document "{self.filename}".
Use the tools to collect the passages you need, one call at a time. When the evidence
covers every part of the question, choose "answer". Search for each part of a
multi-part question separately, and use exact names and numbers in keyword searches.

Tools:
{tools}
- answer: stop searching and answer from the evidence; input is empty

Question: {question}
Steps taken: {sum(s["type"] == "tool" for s in self.steps)} of {self.max_steps}
Evidence count: {len(self.scores)}
Evidence so far:
{evidence}

Previous tool calls:
{history}

Reply as JSON: {{"thought": "what is still missing", "action": "tool name or answer", "input": "tool input"}}"""
        return parse_json(self.llm(prompt, as_json=True))

    def research(self, question: str):
        for _ in range(self.max_steps):
            d = self.plan(question)
            action = str(d.get("action", "")).strip()
            arg = d.get("input", "")
            if action == "answer" and self.scores:
                self.step("think", thought=str(d.get("thought", "")), action="answer")
                return
            if action not in TOOLS:
                action, arg = "search", question  # unusable plan: search for the question itself
            if not str(arg).strip():
                arg = question
            try:
                found = self.tool(action, arg)
            except (ValueError, BackendError) as e:
                found = []
                self.step("tool", tool=action, input=str(arg), found=[], error=str(e),
                          thought=str(d.get("thought", "")))
                continue
            new = [i for i in found if i not in self.scores]
            self.add_evidence(found)
            self.step("tool", tool=action, input=str(arg), found=found, new=new,
                      thought=str(d.get("thought", "")),
                      pages=[page_label(self.chunks[i]) for i in found])

    # ---------- answering ----------

    def answer(self, question: str) -> tuple[str, list[int]]:
        ids = self.sources()
        text = self.llm(rag_prompt(question, [self.chunks[i] for i in ids]), stage="generate_ms")
        return text, ids

    def summarize(self, question: str) -> tuple[str, list[int]]:
        """Map-reduce: summarize groups of chunks spread through the document, then combine."""
        n = len(self.chunks)
        picked = sorted({round(j * (n - 1) / max(1, SUMMARY_CHUNKS - 1)) for j in range(min(n, SUMMARY_CHUNKS))})
        partials = []
        for g in range(0, len(picked), 4):
            group = [self.chunks[i] for i in picked[g:g + 4]]
            listing = "\n\n".join(f"({page_label(c)}) {c['text']}" for c in group)
            partials.append(self.llm(f"""Summarize the passages below in 2-3 sentences, keeping names and numbers.

Passages:
{listing}

Summary:""", stage="generate_ms"))
            self.step("map", pages=[page_label(c) for c in group], summary=partials[-1])
        self.add_evidence(picked)
        joined = "\n".join(f"- {p}" for p in partials)
        text = self.llm(f"""Combine these partial summaries of "{self.filename}" into one answer to the request.

Request: {question}
Partial summaries:
{joined}

Answer:""", stage="generate_ms")
        return text, picked

    # ---------- self-check ----------

    def check(self, answer: str, ids: list[int]) -> dict:
        """Split the answer into claims and ask whether each is backed by the sources it cites."""
        claims = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.split()) >= 4]
        refused = bool(REFUSAL.search(answer))
        if not claims or refused:
            return {"grounded": True, "claims": [], "invalid_citations": [], "refused": refused}
        bad_cites = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)
                            if not 1 <= int(n) <= len(ids)})
        context = "\n\n".join(f"[{n}] {self.chunks[i]['text']}" for n, i in enumerate(ids, 1))
        listing = "\n".join(f"{n}. {c}" for n, c in enumerate(claims, 1))
        data = parse_json(self.llm(f"""Check each claim against the numbered passages. A claim is supported only if
the passages state it or it follows directly from them.

Passages:
{context}

Claims:
{listing}

Reply as JSON: {{"claims": [{{"n": claim number, "supported": true or false}}]}}""",
                                   as_json=True, stage="check_ms"))
        verdict = {}
        for c in data.get("claims", []) if isinstance(data.get("claims"), list) else []:
            if isinstance(c, dict) and isinstance(c.get("n"), int):
                verdict[c["n"]] = bool(c.get("supported"))
        results = [{"claim": c, "supported": verdict.get(n, True)} for n, c in enumerate(claims, 1)]
        return {"grounded": all(r["supported"] for r in results) and not bad_cites,
                "claims": results, "invalid_citations": bad_cites, "refused": False}

    # ---------- run ----------

    def run(self, question: str) -> dict:
        t0 = time.perf_counter()
        route, reason = self.route(question)
        self.step("route", route=route, reason=reason)

        if route == "direct":
            answer = self.llm(plain_prompt(question), stage="generate_ms")
            return self.finish(question, route, answer, [], None, t0)

        if route == "summary":
            answer, ids = self.summarize(question)
            return self.finish(question, route, answer, ids, None, t0)

        if route == "lookup":
            found = self.tool("search", question)
            self.add_evidence(found)
            self.step("tool", tool="search", input=question, found=found, new=found,
                      thought="single lookup", pages=[page_label(self.chunks[i]) for i in found])
            answer, ids = self.answer(question)
            check = self.check(answer, ids) if self.self_check else None
            if check and (check["refused"] or not check["grounded"]):
                why = "the first answer was a refusal" if check["refused"] else "the answer failed the self-check"
                self.step("escalate", reason=why, to="research")
                route = "research"
            else:
                return self.finish(question, route, answer, ids, check, t0)

        self.research(question)
        answer, ids = self.answer(question)
        check = self.check(answer, ids) if self.self_check else None

        # Corrective retrieval: search for the unsupported claims and answer once more
        if check and not check["grounded"] and not check["refused"]:
            missing = [c["claim"] for c in check["claims"] if not c["supported"]]
            if missing:
                found = self.tool("search", " ".join(missing)[:300])
                self.add_evidence(found)
                self.step("correct", claims=missing, found=found,
                          pages=[page_label(self.chunks[i]) for i in found])
                answer, ids = self.answer(question)
                check = self.check(answer, ids)
        return self.finish(question, route, answer, ids, check, t0)

    def finish(self, question, route, answer, ids, check, t0) -> dict:
        if check:
            self.step("check", **check)
        result = {
            "question": question, "route": route, "answer": answer, "sources": ids,
            "citations": [{"n": n, "id": i, "pages": page_label(self.chunks[i]),
                           "text": self.chunks[i]["text"]} for n, i in enumerate(ids, 1)],
            "check": check, "steps": self.steps, "llm_calls": self.llm_calls,
            "timings": {k: round(v, 1) for k, v in self.timings.items()},
            "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        self.emit({"type": "answer", **{k: v for k, v in result.items() if k != "steps"}})
        return result


def run(question: str, chunks: list[dict], filename="the document", emit=None, **kw) -> dict:
    if not chunks:
        raise BackendError("upload a PDF first")
    return Agent(chunks, filename, emit, **kw).run(question)
