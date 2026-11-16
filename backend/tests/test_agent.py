"""The agentic pipeline against the mock model: routing, tool loop, self-check and the /agent stream."""
import os
import json
import pytest
from conftest import HAVE_HNSW
from sample_doc import make_pdf

pytestmark = pytest.mark.skipif(not HAVE_HNSW, reason="hnsw binary not available")


@pytest.fixture(scope="module")
def client(workdir):
    import app as backend
    backend.app.testing = True
    c = backend.app.test_client()
    pdf = make_pdf(os.path.join(workdir, "agent.pdf"))
    with open(pdf, "rb") as f:
        r = c.post("/upload", data={"file": (f, "agent.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return c


@pytest.fixture(scope="module")
def chunks(client):
    from pipeline import load_doc
    return load_doc()["chunks"]


def events(client, **body):
    r = client.post("/agent", json=body)
    assert r.status_code == 200
    return [json.loads(line) for line in r.get_data(as_text=True).splitlines() if line]


def test_lookup_finds_the_fact_and_passes_check(client):
    ev = events(client, question="Who designed the Kestrel bridge in Port Emery?")
    assert ev[0] == {"type": "route", "route": "lookup", "reason": "keyword rules"}
    answer = ev[-1]
    assert answer["type"] == "answer"
    assert "Tomas Ruel" in answer["answer"]
    assert answer["check"]["grounded"] is True
    assert any("Kestrel" in c["text"] for c in answer["citations"])


def test_research_route_runs_the_tool_loop(chunks):
    import agent
    res = agent.run("Who designed the Aurora reactor and the Lumen protocol?", chunks)
    assert res["route"] == "research"
    tools = [s["tool"] for s in res["steps"] if s["type"] == "tool"]
    assert tools[:2] == ["search", "keyword"]
    assert res["sources"] and len(res["sources"]) <= agent.MAX_SOURCES
    assert res["timings"]["plan_ms"] > 0


def test_summary_route_maps_and_reduces(chunks):
    import agent
    res = agent.run("Summarize this document", chunks)
    assert res["route"] == "summary"
    assert sum(s["type"] == "map" for s in res["steps"]) >= 2
    assert res["answer"]


def test_self_check_flags_unsupported_claims(chunks):
    import agent
    a = agent.Agent(chunks)
    a.add_evidence([0])
    ok = a.check(chunks[0]["text"].split(".")[0] + ". [1]", [0])
    assert ok["grounded"]
    bad = a.check("The moon is made of green cheese and orbits Jupiter. [1] [7]", [0])
    assert not bad["grounded"] and bad["invalid_citations"] == [7]
    assert bad["claims"][0]["supported"] is False


def test_unparseable_route_falls_back_to_rules(chunks, monkeypatch):
    import agent
    monkeypatch.setattr(agent, "llm_call", lambda *a, **k: ("not json", 1.0))
    assert agent.Agent(chunks).route("Give me an overview") == ("summary", "fallback rule")
    assert agent.Agent(chunks).route("Compare the dam and the canal")[0] == "research"


def test_agent_rejects_empty_question(client):
    assert client.post("/agent", json={"question": " "}).status_code == 400
