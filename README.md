# DocuMind

Upload PDFs, ask questions, and get streamed answers with page-level citations.
FastAPI + PostgreSQL/pgvector, hybrid search (vectors + full-text, fused with RRF), multi-tenant.

## Architecture (v0.1, local)

```
POST /documents ──► save PDF ──► 202 Accepted
                         └─► background task: pypdf (per page) → chunk (1000 chars, 150 overlap)
                                               → embed (batch) → INSERT chunks (+ tsvector)
POST /chat ──► embed question ─┬─► pgvector cosine top-30 ─┐
                               └─► Postgres full-text top-30 ┴─► RRF fusion → top-5 → LLM → SSE stream
```
Every query filters by `owner_id`, so users only search their own documents.

## Run locally

Prereqs: Docker Desktop and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env               # add your Gemini key (free tier) or use LLM_PROVIDER=fake
docker compose up -d --wait        # Postgres + pgvector on localhost:5433
uv sync                            # create .venv and install deps
uv run pytest -q                   # unit tests
uv run fastapi dev app/main.py     # http://localhost:8000/docs (auto-reload)
```

Try it:
```bash
curl -H 'X-User-Id: vishal' -F file=@some.pdf localhost:8000/documents
curl -H 'X-User-Id: vishal' localhost:8000/documents          # wait until status = ready
curl -N -H 'X-User-Id: vishal' -H 'Content-Type: application/json' \
     -d '{"question":"What are the payment terms?"}' localhost:8000/chat
```

`/chat` streams Server-Sent Events: `sources` (citations), many `token` events, then `done`.

**Changing the embedding model or `EMBED_DIM`** changes the vector column size. Reset the DB with
`docker compose down -v && docker compose up -d --wait`.

## Code map

| File | What it does |
|---|---|
| `app/config.py` | Settings from `.env` (pydantic-settings) |
| `app/db.py` | Schema (documents, chunks, HNSW + GIN indexes), connection pool |
| `app/ingest.py` | PDF text per page, recursive chunker, background ingestion |
| `app/retrieval.py` | Hybrid search SQL with Reciprocal Rank Fusion |
| `app/llm.py` | Embeddings and streaming chat over any OpenAI-compatible API; grounded prompt |
| `app/main.py` | FastAPI routes |

## Roadmap

- [x] Upload, background ingestion, pgvector, hybrid search, streamed answers with citations, tenant isolation
- [ ] JWT auth instead of the `X-User-Id` header
- [ ] OCR for scanned PDFs (Tesseract / a vision LLM)
- [ ] Reranking (bge-reranker or Cohere) over the fused top 30
- [ ] Real job queue (RQ/Celery + Redis) with retries instead of FastAPI BackgroundTasks
- [ ] Conversation memory (follow-up questions)
- [ ] UI (Streamlit or Next.js)
- [ ] Evals: 30–50 Q&A pairs, RAGAS faithfulness / answer relevance
- [ ] Tracing with Langfuse (latency, tokens, cost per request)
- [ ] Dockerfile, deploy to AWS (ECS + RDS + S3), GitHub Actions CI, demo video
