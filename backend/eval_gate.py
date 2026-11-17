"""Quality gate for CI: index a fixed test document, write a test set, run the
evaluation and fail if any score falls below the thresholds in eval_thresholds.json.

    python eval_gate.py --mock                  # self-contained, no models needed (CI default)
    python eval_gate.py                         # against the configured provider (real models)
    python eval_gate.py --pdf my.pdf --n 30     # your own document

With --mock the scores only prove the pipeline works end to end; they say
nothing about how a real model performs. The Markdown report is written to
--report and, on GitHub Actions, to the job summary.
"""
import os
import sys
import json
import shutil
import argparse
import tempfile
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(HERE, "tests")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="start the mock model server and use it")
    ap.add_argument("--pdf", help="document to evaluate on (default: the synthetic test PDF)")
    ap.add_argument("--n", type=int, default=20, help="test questions to write")
    ap.add_argument("--configs", nargs="*", default=["no_rag", "basic", "advanced", "agent"])
    ap.add_argument("--thresholds", default=os.path.join(HERE, "eval_thresholds.json"))
    ap.add_argument("--report", default="eval-report.md")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    workdir = tempfile.mkdtemp(prefix="rag-eval-gate-")
    exe = "hnsw.exe" if os.name == "nt" else "hnsw"
    built = os.environ.get("HNSW_EXE") or os.path.join(ROOT, "hnsw", exe)
    if not os.path.exists(built):
        subprocess.run(["g++", "-O2", "-std=c++17", "-o", built, os.path.join(ROOT, "hnsw", "hnsw.cpp")],
                       check=True)
    shutil.copy(built, os.path.join(workdir, exe))
    os.environ["HNSW_DIR"] = workdir
    os.environ["HNSW_EXE"] = os.path.join(workdir, exe)
    if args.mock:
        import mock_ollama
        server = mock_ollama.serve()
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_URL"] = f"http://127.0.0.1:{server.server_port}"

    import fitz
    import pipeline
    import evaluation
    from sample_doc import make_pdf

    pdf_path = args.pdf or make_pdf(os.path.join(workdir, "gate.pdf"))
    pdf = fitz.open(pdf_path)
    pages = pipeline.extract_pages(pdf)
    chunks = pipeline.chunk_pages(pages)
    vectors = pipeline.embed([c["text"] for c in chunks], "document")
    pipeline.run_hnsw(["reset"])
    pipeline.run_hnsw(["build"], stdin="".join(f"{c['id']}\t{pipeline.vec_to_str(v)}\n"
                                                for c, v in zip(chunks, vectors)))
    with open(pipeline.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump({"filename": os.path.basename(pdf_path), "pages": len(pages), "chunks": chunks}, f)

    evaluation.generate_dataset(n=args.n, workers=args.workers)
    run = evaluation.run_evaluation(config_ids=args.configs, workers=args.workers, b=500)
    report = evaluation.markdown_report(run)

    with open(args.thresholds, encoding="utf-8") as f:
        thresholds = json.load(f)
    failures, rows = [], []
    for cid, limits in thresholds.items():
        if cid not in run["summary"]["configs"]:
            continue
        got = run["summary"]["configs"][cid]
        for key, limit in limits.items():
            value = got["latency"]["p95"] if key == "p95_ms" else got.get(key, {}).get("mean")
            ok = value is not None and (value <= limit if key in ("refusal", "p95_ms") else value >= limit)
            rows.append(f"| {cid} | {key} | {'—' if value is None else f'{value:.3f}'} | "
                        f"{'≤' if key in ('refusal', 'p95_ms') else '≥'} {limit} | {'✅' if ok else '❌'} |")
            if not ok:
                failures.append(f"{cid}.{key} = {value} (limit {limit})")

    gate = ["## Quality gate", "", "Model: " + ("mock (pipeline check only)" if args.mock else
                                             f"{pipeline.PROVIDER} {pipeline.LLM_MODEL}"), "",
            "| Config | Metric | Value | Limit | |", "|---|---|---:|---|---|", *rows, ""]
    full = "\n".join(gate) + "\n" + report
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(full)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(full)
    print("\n".join(gate))
    if failures:
        print("Quality gate failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("Quality gate passed.")


if __name__ == "__main__":
    main()
