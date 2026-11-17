# DocuMind

**Chat with your documents.** Upload PDFs, scans or photos of invoices, ask questions in plain English, and get
streamed answers with page-level citations, or an honest "I don't know".

FastAPI · PostgreSQL + pgvector · hybrid search + cross-encoder reranking · OCR · Redis/RQ workers · JWT auth ·
Streamlit · LLM evals · Langfuse tracing · Docker · GitHub Actions

<!-- After deploying, add: Live demo link · 2-minute demo video link · screenshot of the UI -->

## Features

- **Grounded answers with citations.** Every claim is tagged `[S1]`, `[S2]` and linked to a file and page.
  Answers that cite sources which weren't retrieved are flagged.
- **Hybrid retrieval.** pgvector cosine search and Postgres full-text search are fused with Reciprocal Rank
  Fusion, then reranked by a local cross-encoder (ONNX on CPU, no API cost).
- **OCR.** Scanned PDF pages and PNG/JPG uploads are read with Tesseract or a vision LLM.
- **Background ingestion.** Uploads return `202` at once; a Redis/RQ worker extracts, chunks and embeds, with
  3 retries and backoff. Re-running a job is safe (chunks are replaced in one transaction).
- **Conversations.** Follow-ups like "and for managers?" are rewritten into standalone search queries using
  the chat history.
- **Multi-tenant.** JWT auth (Argon2id password hashes), and every query is scoped to the signed-in user.
- **Observability.** Latency, time to first token, tokens and cost are stored per answer, with a usage
  dashboard (avg and p95 latency, total cost). Optional Langfuse traces of every step.
- **Evals.** 34-question test set over a sample corpus: hit@k, MRR, LLM-judged correctness and faithfulness,
  refusal accuracy on unanswerable questions, latency and cost.

## Architecture

```
                      ┌────────────── Streamlit UI (:8501) ──────────────┐
                      │ login · upload · documents · chat · usage        │
                      └──────────────────────┬───────────────────────────┘
                                             │ HTTP + SSE, Bearer JWT
┌────────────────────────────────── FastAPI (:8000) ───────────────────────────────────┐
│ /auth   register, login (OAuth2 password flow) → JWT                                 │
│ /documents  upload → storage (disk/S3) → enqueue ──────────────► Redis ──► RQ worker │
│ /chat   history → rewrite follow-up → hybrid search → rerank → LLM → SSE stream      │
│ /conversations · /stats                                                              │
└─────────────┬──────────────────────────────────────────────────────────────┬─────────┘
              │                                                              │
   PostgreSQL 17 + pgvector                                    Worker: read file
   users · documents · chunks (vector + tsvector)              → text layer, or OCR for scans
   conversations · messages (sources, usage)                   → chunk (1000 chars, 150 overlap)
                                                               → embed (batched) → chunks table
```

**Query pipeline**

```
question ─► rewrite with history ─► embed ─┬─► pgvector cosine, top 30 ───┐
                                           └─► full-text search, top 30 ──┴─► RRF fusion, top 20
          ─► cross-encoder rerank, top 5 ─► grounded prompt ─► LLM stream ─► citation check ─► save + usage
```

## Run locally

**Prerequisites:** Docker Desktop and [uv](https://docs.astral.sh/uv/). You also need an LLM key; Gemini's
free tier works (<https://aistudio.google.com/apikey>), or set `LLM_PROVIDER=fake` to run with no key.

```bash
cp .env.example .env     # set LLM_API_KEY and JWT_SECRET
```

**Option A: everything in Docker** (closest to production)
```bash
docker compose up -d --build
open http://localhost:8501          # UI      (API docs: http://localhost:8000/docs)
docker compose logs -f api worker   # watch ingestion and per-answer latency/cost
```

**Option B: develop with hot reload** (infrastructure in Docker, Python processes local)
```bash
docker compose up -d db redis
uv sync
uv run fastapi dev app/main.py                 # terminal 1: API on :8000
uv run python -m app.worker                    # terminal 2: ingestion worker
uv run streamlit run ui/streamlit_app.py       # terminal 3: UI on :8501
```
On macOS, install Tesseract for local OCR (`brew install tesseract`), or set `OCR_ENGINE=llm` to use the
vision model. The Docker image already includes it.

**Tests and lint**
```bash
uv run pytest -q          # 34 tests; uses a separate documind_test database, no API calls
uv run ruff check . && uv run ruff format --check .
```

## Evaluation

```bash
uv run python -m evals.make_corpus   # sample handbook, invoice and a scanned notice
uv run python -m evals.run           # writes evals/results/latest.md
```

| Metric | What it measures |
|---|---|
| hit@1, hit@5, MRR | Retrieval: is the correct file and page in the top results |
| correctness | LLM judge: does the answer contain the reference facts |
| faithfulness | LLM judge: is every claim supported by the retrieved sources (no hallucination) |
| refusal accuracy | Says "I don't know" on 4 questions the documents can't answer |
| citation rate, invalid citations | Answers cite their sources, and only sources that exist |
| p50/p95 latency, cost | Speed and price per question |

<!-- Paste the summary table from evals/results/latest.md here after your first run. -->

Use `--judge-model` to judge with a stronger model than the one being judged, `--no-judge` for a cheap
retrieval-only run, and change one setting at a time (`RERANKER=none`, `TOP_K=3`, `CHUNK_SIZE=600`) to see what
each part contributes.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create an account (`email`, `password` ≥ 8 chars) |
| POST | `/auth/login` | Form fields `username`, `password` → `access_token` |
| GET | `/auth/me` | Current user |
| POST | `/documents` | Upload PDF/PNG/JPG (multipart `file`) → `202`, ingestion queued |
| GET | `/documents` · `/documents/{id}` | List / status (`queued`, `processing`, `ready`, `failed`) |
| GET | `/documents/{id}/file` | Download the original file |
| POST | `/documents/{id}/reprocess` | Re-run ingestion |
| DELETE | `/documents/{id}` | Delete the document, its chunks and its file |
| POST | `/chat` | `{question, conversation_id?, document_ids?}` → SSE: `meta`, `sources`, `token`…, `done` |
| GET/DELETE | `/conversations` · `/conversations/{id}` | Chat history with sources and usage |
| GET | `/stats` | Documents, answers, tokens, cost, avg and p95 latency |

Interactive docs at `/docs`. Click **Authorize** and sign in to try the protected routes.

## Design decisions

- **Postgres for vectors, keywords and app data.** One database, transactional re-ingestion, tenant filtering
  in the same query as the vector search. A dedicated vector DB only pays off at much larger scale.
- **Hybrid search and RRF.** Embeddings miss exact tokens such as invoice numbers and GSTINs; full-text search
  misses paraphrases. RRF combines rankings without having to calibrate two different score scales.
- **Rerank only the top 20.** A cross-encoder is much more accurate but too slow to run on every chunk.
- **Per-page chunking.** Chunks never span pages, so every citation points at one exact page.
- **The prompt treats document text as data.** Sources go inside `<sources>` tags with an instruction to
  ignore instructions found in them; a basic defence against prompt injection from uploaded files.
- **A provider-agnostic LLM layer.** Any OpenAI-compatible API (Gemini, OpenAI, Ollama). Models are config,
  not code.

## Project layout

| Path | What it does |
|---|---|
| `app/main.py` | App setup, startup checks, `/health` |
| `app/auth.py` | Register, login, JWT issue and verify, `current_user` dependency |
| `app/routers/documents.py` | Upload (type detection by file bytes), list, download, reprocess, delete |
| `app/routers/chat.py` | SSE chat, conversation memory, usage tracking, `/conversations`, `/stats` |
| `app/ingest.py` | Text extraction, OCR fallback, recursive chunker, idempotent ingestion |
| `app/ocr.py` | Tesseract or vision-LLM OCR |
| `app/retrieval.py` | Hybrid search SQL with RRF, then rerank |
| `app/rerank.py` | Local cross-encoder (fastembed / ONNX) |
| `app/llm.py` | Embeddings, streaming answers, follow-up rewriting, vision OCR |
| `app/jobs.py`, `app/worker.py` | Enqueueing and the RQ worker with retries |
| `app/storage.py` | Local disk or S3 |
| `app/observability.py` | Optional Langfuse tracing |
| `ui/streamlit_app.py` | Web UI |
| `evals/` | Corpus generator, dataset, eval runner with LLM judge |
| `docs/DEPLOY_AWS.md` | Deploying to AWS (single EC2, or ECS + RDS + S3) |

## Roadmap

- [x] Ingestion, pgvector, hybrid search, streamed answers with citations, tenant isolation
- [x] JWT auth
- [x] OCR for scanned PDFs and images
- [x] Cross-encoder reranking
- [x] Redis/RQ job queue with retries
- [x] Conversation memory with follow-up rewriting
- [x] Streamlit UI with a usage dashboard
- [x] Evals (retrieval metrics + LLM judge)
- [x] Tracing (Langfuse) and per-answer latency, tokens and cost
- [x] Docker, docker compose, GitHub Actions CI, AWS deploy guide
- [ ] Deploy and record the demo video
- [ ] Semantic cache for repeated questions (Redis)
- [ ] PII masking before text is sent to the LLM
