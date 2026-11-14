"""Upload a PDF, write a test set and run the whole evaluation against the mock Ollama."""
import os
import time
import pytest
from conftest import HAVE_HNSW
from sample_doc import make_pdf

pytestmark = pytest.mark.skipif(not HAVE_HNSW, reason="hnsw binary not available")


@pytest.fixture(scope="module")
def client(workdir):
    import app as backend
    backend.app.testing = True
    c = backend.app.test_client()
    pdf = make_pdf(os.path.join(workdir, "sample.pdf"))
    with open(pdf, "rb") as f:
        r = c.post("/upload", data={"file": (f, "sample.pdf")}, content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    return c


def wait(client, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        job = client.get("/eval/job").get_json()
        if job["status"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_full_evaluation(client):
    r = client.post("/eval/run", json={})
    assert r.status_code == 400  # no test set yet

    assert client.post("/eval/dataset", json={"n": 8, "workers": 2}).get_json()["started"]
    job = wait(client)
    assert job["status"] == "done", job
    ds = client.get("/eval/dataset").get_json()
    assert ds["current"] and len(ds["items"]) == 8
    assert all(i["relevant"] and i["source_chunk"] in i["relevant"] for i in ds["items"])

    # Uploads are refused while a job runs; deleting a question works between jobs
    assert client.delete(f"/eval/dataset/{ds['items'][-1]['id']}").status_code == 200

    assert client.post("/eval/run", json={"workers": 2}).get_json()["started"]
    job = wait(client)
    assert job["status"] == "done", job
    runs = client.get("/eval/runs").get_json()["runs"]
    run = client.get(f"/eval/runs/{runs[0]['id']}").get_json()

    assert len(run["configs"]) == 10 and len(run["items"]) == 7
    assert len(run["records"]) == 70
    s = run["summary"]
    basic, exact = s["configs"]["basic"], s["configs"]["exact"]
    # The mock's hashed embeddings make retrieval easy: exact search finds the evidence
    assert exact["hit5"]["mean"] >= 0.8
    assert basic["hit5"]["mean"] >= 0.5
    assert "hit5" not in s["configs"]["no_rag"]
    assert s["configs"]["no_rag"]["refusal"]["mean"] == 1.0
    assert s["configs"]["basic"]["correctness"]["mean"] > s["configs"]["no_rag"]["correctness"]["mean"]
    assert set(s["effects"]) == {"rewrite", "hybrid", "rerank"}
    assert all(len(e["pairs"]) == 4 for e in s["effects"].values())
    lat = basic["latency"]
    assert lat["p50"] > 0 and lat["stages"]["generate_ms"] > 0

    md = client.get(f"/eval/runs/{run['id']}/report.md").get_data(as_text=True)
    assert "## What each advanced step adds" in md and "Basic RAG (HNSW)" in md


def test_subset_without_judge(client):
    assert client.post("/eval/run", json={"configs": ["basic", "hybrid"], "judge": False,
                                          "limit": 3}).get_json()["started"]
    job = wait(client)
    assert job["status"] == "done", job
    run = client.get(f"/eval/runs/{job['result_id']}").get_json()
    assert [c["id"] for c in run["configs"]] == ["basic", "hybrid"]
    assert "correctness" not in run["summary"]["configs"]["basic"]
    assert list(run["summary"]["effects"]) == ["hybrid"]
