#!/bin/sh
# Start Ollama, wait until it answers, then serve the app on $PORT.
set -e
mkdir -p "$HNSW_DIR"
ollama serve &
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:11434/api/version > /dev/null && break
  sleep 1
done
exec gunicorn --bind 0.0.0.0:"${PORT:-7860}" --workers 1 --threads 8 --timeout 900 app:app
