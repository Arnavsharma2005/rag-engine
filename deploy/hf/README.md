---
title: RAG Engine
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Agentic RAG over your PDF with a C++ HNSW index
---

# RAG Engine

Upload a PDF and compare answers with no RAG, basic RAG, advanced RAG and an agent that routes,
searches with tools and checks its own citations. Ollama runs inside this Space, so documents stay here.

This Space is deployed automatically from GitHub by CI. Source, architecture and evaluation results:
https://github.com/Arnavsharma2005/rag-engine

Notes on the free tier: answers come from a small model (llama3.2:1b) on CPU, so expect a few seconds
per answer. The Space sleeps after 48 hours without visitors, and the indexed PDF is lost on restart.
Everyone using the Space shares one index, so don't upload private documents to a public Space.
