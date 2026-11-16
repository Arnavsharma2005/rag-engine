"""Push the app to a Hugging Face Space (Docker SDK), creating the Space if needed.

    HF_TOKEN=... python deploy/hf/deploy.py owner/space-name [--cors https://owner.github.io]

Uploads only what the image needs, with deploy/hf/Dockerfile and README.md at the
Space's root. The Space then rebuilds itself.
"""
import os
import sys
import shutil
import argparse
import tempfile
from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IGNORE = shutil.ignore_patterns("node_modules", "dist", "__pycache__", "*.pyc", "eval", "tests",
                                ".pytest_cache", "chunks.json", "index.bin", "hnsw", "hnsw.exe", "eval-report.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("space", help="owner/name of the Space")
    ap.add_argument("--cors", help="extra allowed origin, e.g. the GitHub Pages URL")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--message", default="Deploy from GitHub Actions")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or sys.exit("HF_TOKEN is not set")

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    if args.cors:
        api.add_space_variable(args.space, "CORS_ORIGINS", args.cors)

    with tempfile.TemporaryDirectory() as stage:
        shutil.copytree(os.path.join(ROOT, "backend"), os.path.join(stage, "backend"), ignore=IGNORE)
        shutil.copytree(os.path.join(ROOT, "frontend"), os.path.join(stage, "frontend"), ignore=IGNORE)
        os.makedirs(os.path.join(stage, "hnsw"))
        shutil.copy(os.path.join(ROOT, "hnsw", "hnsw.cpp"), os.path.join(stage, "hnsw"))
        os.makedirs(os.path.join(stage, "deploy", "hf"))
        shutil.copy(os.path.join(ROOT, "deploy", "hf", "start.sh"), os.path.join(stage, "deploy", "hf"))
        shutil.copy(os.path.join(ROOT, "deploy", "hf", "Dockerfile"), os.path.join(stage, "Dockerfile"))
        shutil.copy(os.path.join(ROOT, "deploy", "hf", "README.md"), os.path.join(stage, "README.md"))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=stage,
                          commit_message=args.message, delete_patterns=["*"])
    print(f"Deployed to https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
