# Hugging Face Space image: everything in one container on the free CPU tier.
# Ollama runs inside the container with small models baked in at build time,
# so no API keys are needed and documents never leave the Space.

FROM gcc:14 AS hnsw
WORKDIR /src
COPY hnsw/hnsw.cpp .
RUN g++ -O2 -std=c++17 -static -o hnsw hnsw.cpp

FROM node:22-slim AS ui
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN VITE_API_URL= npm run build

FROM ollama/ollama:0.40.16 AS ollama

FROM python:3.13-slim
# llama3.2:1b answers in seconds on 2 free vCPUs; llama3.2 (3B) is better but several times slower
ARG LLM_MODEL=llama3.2:1b
ARG EMBED_MODEL=nomic-embed-text
# CI sets PULL_MODELS=0 to check the image builds without downloading ~1.6 GB of weights
ARG PULL_MODELS=1
ENV PYTHONUNBUFFERED=1 \
    HNSW_EXE=/app/hnsw/hnsw \
    HNSW_DIR=/home/user/data \
    STATIC_DIR=/app/static \
    OLLAMA_MODELS=/models \
    OLLAMA_HOST=127.0.0.1:11434 \
    OLLAMA_URL=http://127.0.0.1:11434 \
    OLLAMA_NUM_PARALLEL=2 \
    LLM_MODEL=${LLM_MODEL} \
    EMBED_MODEL=${EMBED_MODEL} \
    PORT=7860

COPY --from=ollama /usr/bin/ollama /usr/bin/ollama
COPY --from=ollama /usr/lib/ollama /usr/lib/ollama
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models && if [ "$PULL_MODELS" = "1" ]; then \
      (ollama serve > /tmp/ollama.log 2>&1 &) && \
      for i in $(seq 1 30); do curl -sf http://127.0.0.1:11434/api/version > /dev/null && break; sleep 1; done && \
      ollama pull "$LLM_MODEL" && ollama pull "$EMBED_MODEL"; \
    fi && chmod -R a+rX /models

WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0
COPY backend/ .
COPY --from=hnsw /src/hnsw /app/hnsw/hnsw
COPY --from=ui /src/dist /app/static
COPY deploy/hf/start.sh /app/start.sh

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user && chmod +x /app/start.sh
USER user
EXPOSE 7860
CMD ["/app/start.sh"]
