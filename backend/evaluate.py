"""Run the evaluation harness from the command line, for long runs without the UI.

    python evaluate.py generate --n 40          # write a test set from the indexed PDF
    python evaluate.py run                      # every configuration, with the LLM judge
    python evaluate.py run --configs basic advanced --no-judge --limit 10
    python evaluate.py report 20261114-101500   # print a saved run as Markdown

Results are saved in hnsw/eval/runs/ and show up in the UI's Evaluate tab.
"""
import sys
import argparse
import evaluation


def show(done, total, message):
    width = 30
    filled = int(width * done / total) if total else 0
    sys.stderr.write(f"\r[{'#' * filled}{'.' * (width - filled)}] {done}/{total} {message[:70]:<70}")
    sys.stderr.flush()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write a test set from the indexed PDF")
    g.add_argument("--n", type=int, default=25, help="number of questions (default 25)")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--workers", type=int, default=1, help="parallel LLM calls")
    r = sub.add_parser("run", help="evaluate every configuration on the test set")
    r.add_argument("--configs", nargs="*", choices=list(evaluation.CONFIG_BY_ID),
                   help="default: all of them")
    r.add_argument("--no-judge", action="store_true", help="skip the LLM judge (much faster)")
    r.add_argument("--limit", type=int, help="only the first N questions")
    r.add_argument("--workers", type=int, default=1,
                   help="questions in parallel; match OLLAMA_NUM_PARALLEL")
    r.add_argument("--out", help="also write the Markdown report to this file")
    rep = sub.add_parser("report", help="print a saved run as Markdown")
    rep.add_argument("run_id", nargs="?", help="default: the latest run")
    args = p.parse_args()

    if args.cmd == "generate":
        ds = evaluation.generate_dataset(n=args.n, seed=args.seed, workers=args.workers, progress=show)
        sys.stderr.write("\n")
        print(f"{len(ds['items'])} questions written from {ds['doc']['filename']} "
              f"({ds['stats']['attempted']} chunks tried, rejected: {ds['stats']['rejected'] or 'none'})")
        print(f"Saved to {evaluation.DATASET_FILE}")
    elif args.cmd == "run":
        run = evaluation.run_evaluation(config_ids=args.configs, judge=not args.no_judge,
                                        limit=args.limit, workers=args.workers, progress=show)
        sys.stderr.write("\n")
        report = evaluation.markdown_report(run)
        print(report)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(report)
    else:
        run_id = args.run_id or next((r["id"] for r in evaluation.list_runs()), None)
        run = evaluation.load_run(run_id) if run_id else {}
        if not run:
            sys.exit("no saved run found")
        print(evaluation.markdown_report(run))


if __name__ == "__main__":
    main()
