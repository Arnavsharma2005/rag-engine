# One image: C++ HNSW engine + Flask API + built React UI on the same origin.
# Models come from Ollama (OLLAMA_URL) or any OpenAI-compatible API (LLM_PROVIDER=openai).

FROM gcc:14 AS hnsw
WORKDIR /src
COPY hnsw/hnsw.cpp .
RUN g++ -O2 -std=c++17 -static -o hnsw hnsw.cpp

FROM node:22-slim AS ui
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
# Empty API URL: the UI calls the backend on its own origin
RUN VITE_API_URL= npm run build

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    HNSW_EXE=/app/hnsw/hnsw \
    HNSW_DIR=/data \
    STATIC_DIR=/app/static \
    PORT=7860
WORKDIR /app/backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0
COPY backend/ .
COPY --from=hnsw /src/hnsw /app/hnsw/hnsw
COPY --from=ui /src/dist /app/static
RUN useradd -m -u 1000 app && mkdir -p /data && chown app /data
USER app
VOLUME /data
EXPOSE 7860
# One worker: uploads and searches share one on-disk index guarded by an in-process lock.
# Threads keep streaming answers concurrent.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 600 app:app
