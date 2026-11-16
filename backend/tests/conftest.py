"""Point the backend at a mock Ollama and a scratch index before it is imported."""
import os
import sys
import shutil
import subprocess
import tempfile
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
ROOT = os.path.dirname(BACKEND)
sys.path[:0] = [BACKEND, HERE]

import mock_ollama  # noqa: E402

_server = mock_ollama.serve()
_workdir = tempfile.mkdtemp(prefix="rag-eval-test-")
os.environ["OLLAMA_URL"] = f"http://127.0.0.1:{_server.server_port}"
os.environ["HNSW_DIR"] = _workdir
os.environ.pop("JUDGE_MODEL", None)

exe = "hnsw.exe" if os.name == "nt" else "hnsw"
built = os.path.join(ROOT, "hnsw", exe)
target = os.path.join(_workdir, exe)
if os.path.exists(built):
    shutil.copy(built, target)
elif shutil.which("g++"):
    subprocess.run(["g++", "-O2", "-std=c++17", "-o", target, os.path.join(ROOT, "hnsw", "hnsw.cpp")],
                   check=True)
HAVE_HNSW = os.path.exists(target)


@pytest.fixture(scope="session")
def workdir():
    return _workdir
