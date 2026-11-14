import random
import evaluation as ev


def test_retrieval_metrics_ranks():
    m = ev.retrieval_metrics([4, 9, 2, 7, 1], {2, 3})
    assert m["rank"] == 3
    assert (m["hit1"], m["hit3"], m["hit5"]) == (0.0, 1.0, 1.0)
    assert abs(m["mrr"] - 1 / 3) < 1e-9
    assert m["recall5"] == 0.5
    assert 0 < m["ndcg5"] < 1


def test_retrieval_metrics_miss_and_pool():
    m = ev.retrieval_metrics([1, 2, 3, 4, 5], {9}, pool=[1, 2, 3, 4, 5, 9])
    assert m["rank"] is None and m["mrr"] == 0 and m["hit5"] == 0
    assert m["pool_hit"] == 1.0


def test_perfect_ndcg():
    assert ev.retrieval_metrics([3, 1, 2], {3})["ndcg5"] == 1.0


def test_token_f1_and_refusal():
    assert ev.token_f1("finished in 1962", "Construction finished in 1962") > 0.6
    assert ev.token_f1("banana", "apple") == 0
    m = ev.answer_metrics("I don't know based on this document.", "1962", 5)
    assert m["refusal"] == 1.0 and m["cited"] == 0.0 and m["citation_valid"] is None
    m = ev.answer_metrics("It finished in 1962 [2][7].", "1962", 5)
    assert m["cited"] == 1.0 and m["citation_valid"] == 0.0


def test_judge_score_scaling():
    assert ev.judge_score({"x": 5}, "x") == 1.0
    assert ev.judge_score({"x": "1"}, "x") == 0.0
    assert ev.judge_score({"x": 9}, "x") == 1.0
    assert ev.judge_score({}, "x") is None


def test_parse_json_with_noise():
    assert ev.parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert ev.parse_json("nope") == {}


def test_bootstrap_interval_contains_mean():
    vals = [random.Random(1).random() for _ in range(50)]
    b = ev.bootstrap(vals, b=500)
    assert b["lo"] <= b["mean"] <= b["hi"]
    assert ev.bootstrap([1.0] * 20)["p"] < 0.05
    assert ev.bootstrap([0.0] * 20)["p"] == 1.0


def chunk(i, text):
    return {"id": i, "text": text, "page_start": 1, "page_end": 1}


def test_validate_qa_marks_overlapping_chunks_relevant():
    span = "the bridge opened to traffic in 1962 after a long delay"
    chunks = [chunk(0, "intro words " + span), chunk(1, span + " more words"), chunk(2, "other")]
    item, reason = ev.validate_qa({"question": "When did the Kestrel bridge open?", "answer": "1962",
                                   "evidence": span}, chunks[0], chunks)
    assert reason == "" and item["relevant"] == [0, 1] and item["exact_evidence"]


def test_validate_qa_rejects():
    c = chunk(0, "alpha beta gamma delta epsilon")
    assert ev.validate_qa({"question": "What does the passage say about alpha?", "answer": "x",
                           "evidence": "alpha beta"}, c, [c])[0] is None
    assert ev.validate_qa({"question": "What is the speed of omega particles?", "answer": "fast",
                           "evidence": "omega particles are fast"}, c, [c])[1] == "evidence not found in passage"


def test_main_effects_from_factorial_grid():
    # Hybrid always adds +1 hit, the other steps add nothing
    cids = ["basic"] + [c["id"] for c in ev.CONFIGS if c["kind"] == "advanced"]
    items = [{"id": i} for i in range(10)]
    records = []
    for cid in cids:
        hybrid = ev.CONFIG_BY_ID[cid]["flags"]["hybrid"]
        for i in range(10):
            records.append({"config": cid, "item": i, "hit5": 1.0 if hybrid else 0.0,
                            "latency": {**{s: 1.0 for s in ev.STAGES}, "total_ms": 5.0},
                            "total_ms": 5.0})
    s = ev.summarize(records, items, cids, b=200)
    assert len(s["effects"]["hybrid"]["pairs"]) == 4
    assert s["effects"]["hybrid"]["hit5"]["mean"] == 1.0
    assert s["effects"]["rerank"]["hit5"]["mean"] == 0.0
    assert s["deltas"]["hybrid"]["hit5"]["mean"] == 1.0 and s["deltas"]["hybrid"]["hit5"]["p"] < 0.05


def test_sample_spreads_over_document():
    chunks = [chunk(i, " ".join(["word"] * 60)) for i in range(100)]
    picked = ev.sample_chunks(chunks, 10, seed=1)[:10]
    assert sorted(p // 10 for p in picked) == list(range(10))
