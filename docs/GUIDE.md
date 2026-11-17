# DocuMind: The Complete Guide (in simple words)

This one document has everything you need:

1. [What DocuMind is](#1-what-documind-is)
2. [How it works, step by step](#2-how-it-works-step-by-step)
3. [Every concept explained simply](#3-every-concept-explained-simply)
4. [Where everything lives in the code](#4-where-everything-lives-in-the-code)
5. [Things you need to set up (API keys, accounts)](#5-things-you-need-to-set-up)
6. [Run it on your Mac](#6-run-it-on-your-mac)
7. [Tests and evals](#7-tests-and-evals)
8. [Push to GitHub in clean batches](#8-push-to-github-in-clean-batches)
9. [Deploy to AWS (free tier), end to end](#9-deploy-to-aws-free-tier-end-to-end)
10. [Interview questions you should be ready for](#10-interview-questions-and-answers)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What DocuMind is

DocuMind is a **"chat with your documents"** web app.

- You sign up and upload PDFs, scanned PDFs or photos (PNG/JPG), for example a company handbook or an invoice.
- You ask questions in normal English: *"How many leave days do new employees get?"*
- It answers using **only your documents**, and every sentence points to the exact file and page it came from,
  like `[S1] handbook.pdf, page 3`.
- If the answer isn't in your documents, it says **"I don't know"** instead of making something up.

In AI-engineering language, this is a **RAG app** (Retrieval-Augmented Generation): first *retrieve* the right
pieces of your documents, then let the LLM *generate* an answer from just those pieces.

**Why companies care:** every company has PDFs nobody reads (policies, contracts, invoices, manuals). This is
one of the most common AI products being built right now, which is why it's Project 1.

---

## 2. How it works, step by step

There are two flows: **uploading** a document and **asking** a question.

### Flow A: Uploading a document

```
You upload a PDF
   │
   ▼
API saves the file (disk or S3), creates a row in "documents" with status "queued",
and replies 202 Accepted right away (you don't wait)
   │
   ▼
A job is put in a queue (Redis)
   │
   ▼
The worker (a separate program) picks up the job:
   1. Reads text from each page
   2. If a page has (almost) no text, it's a scan → run OCR (Tesseract) to read the image
   3. Cuts the text into chunks of ~1000 characters (with 150 characters of overlap)
   4. Turns every chunk into an embedding (a list of 768 numbers) using Gemini
   5. Saves chunks + embeddings in Postgres
   6. Sets status to "ready"
   (if something fails, it retries up to 3 times with waits of 10s, 30s, 90s, then marks "failed")
```

### Flow B: Asking a question

```
You ask: "and for managers?"
   │
   ▼
1. Rewrite: using the chat history, turn it into a full question:
   "How many leave days do managers get?"
2. Search two ways at the same time:
     a. Meaning search (vector): find chunks whose embedding is close to the question's embedding
     b. Keyword search (full-text): find chunks containing the exact words
3. Merge both result lists (RRF), keep the top 20
4. Rerank: a small local AI model reads question + each chunk and scores how well it answers, keep the top 5
5. Send the 5 chunks + the question to the LLM with strict rules:
   "answer only from these sources, cite them as [S1], [S2], say I don't know otherwise"
6. Stream the answer back word by word (like ChatGPT)
7. Check citations (did it cite a source that doesn't exist?), save the message,
   tokens used, cost and time taken
```

---

## 3. Every concept explained simply

| Concept | Simple explanation | Where in DocuMind |
|---|---|---|
| **LLM** | The AI model that writes text (Gemini, GPT). It only knows what it was trained on plus what you put in the prompt. | `app/llm.py` |
| **RAG** | Give the LLM the right pages from *your* documents inside the prompt, so it answers from facts instead of memory. | The whole app |
| **Hallucination** | When the LLM confidently makes things up. RAG + "say I don't know" + citations reduce this. | `SYSTEM_PROMPT` in `app/llm.py` |
| **Chunk** | A small piece of a document (~1000 characters). We search chunks, not whole documents, because the LLM only needs the relevant part. | `chunk_text()` in `app/ingest.py` |
| **Chunk overlap** | Each chunk repeats the last 150 characters of the previous one, so a sentence cut in half still appears whole in one chunk. | `CHUNK_OVERLAP` |
| **Embedding** | A list of numbers (768 here) that represents the *meaning* of text. Texts with similar meaning have similar numbers, even with different words ("vacation" ≈ "leave"). | `embed()` in `app/llm.py` |
| **Vector database / pgvector** | A database that can quickly find "the closest embeddings" to a question. pgvector adds this to normal Postgres. | `app/db.py`, `chunks.embedding` |
| **Cosine similarity** | The math for "how close are two embeddings". You only need the idea: closer = more similar meaning. | `<=>` operator in SQL |
| **HNSW index** | A shortcut structure so vector search stays fast with millions of chunks (like an index in a book). | `app/db.py` |
| **Full-text search** | Classic keyword search inside Postgres. Great for exact things embeddings miss: invoice numbers, GST numbers, names. | `tsv` column, `app/retrieval.py` |
| **Hybrid search** | Do both vector and keyword search and combine them. Best of both. | `app/retrieval.py` |
| **RRF (Reciprocal Rank Fusion)** | A simple way to merge two ranked lists: each chunk gets `1/(60 + its rank)` from each list, then add up. Chunks high on both lists win. No need to compare different score types. | `HYBRID_SQL` |
| **Reranker (cross-encoder)** | A small, accurate AI model that reads the question and a chunk *together* and scores the match. Too slow for all chunks, so we use it only on the top 20. Runs on CPU, free. | `app/rerank.py` |
| **OCR** | Reading text from an image (a scan or a photo). We use Tesseract (free, local), or the vision LLM as backup. | `app/ocr.py` |
| **Citations** | Each source is labelled `[S1]`, `[S2]`… The LLM must put the label after each fact. We check it only used labels that exist. | `cited_tags()` in `app/llm.py` |
| **Prompt injection** | A document could contain text like "ignore your rules". We put documents inside `<sources>` tags and tell the LLM it's data, not instructions. | `SYSTEM_PROMPT` |
| **Queue + worker** | The API doesn't do slow work itself. It puts a "job" in Redis; a separate worker program does it. The API stays fast, and jobs survive restarts. | `app/jobs.py`, `app/worker.py` |
| **Redis / RQ** | Redis is a fast in-memory store used as the job queue. RQ is a Python library that runs jobs from it. | `app/worker.py` |
| **Retries with backoff** | If a job fails (e.g. Gemini is busy), try again after 10s, then 30s, then 90s. | `Retry(...)` in `app/jobs.py` |
| **Idempotent** | Running the same job twice gives the same result (no duplicate chunks). We delete old chunks and insert new ones in one transaction. | `ingest_document()` |
| **JWT** | After login you get a signed token. The browser sends it with every request (`Authorization: Bearer ...`). The server checks the signature, no session table needed. | `app/auth.py` |
| **Argon2id** | A slow-on-purpose password hashing method, so stolen hashes are hard to crack. We never store real passwords. | `app/auth.py` |
| **Multi-tenant** | Many users, one database. Every query has `WHERE owner_id = <you>`, so nobody sees anyone else's documents. Others' documents return 404 (not 403) so we don't even reveal they exist. | Every SQL query |
| **SSE (Server-Sent Events)** | A simple way for the server to send the answer in small pieces as it's generated, so you see words appear live. | `app/routers/chat.py` |
| **Conversation memory** | We store chat history. For follow-up questions, the LLM rewrites them into a full question before searching. | `condense_question()` |
| **Tokens and cost** | LLMs charge per token (~4 characters). We record tokens and cost (USD) for every answer. | `usage` in `messages` table |
| **p95 latency** | 95% of answers are faster than this number. Better than average for showing real user experience. | `/stats` |
| **Evals** | Automated tests for AI quality: a list of questions with known answers, run through the app, scored. | `evals/` |
| **LLM-as-judge** | Using an LLM to grade answers ("is this correct?", "is every claim supported by the sources?"). | `evals/run.py` |
| **Langfuse** | Optional tracing tool. Shows every step (search, rerank, LLM call) with timings and cost in a web dashboard. | `app/observability.py` |
| **Docker / docker compose** | Packages the app and everything it needs into containers. One command starts DB, Redis, API, worker and UI. | `Dockerfile`, `docker-compose.yml` |
| **CI (GitHub Actions)** | Every push to GitHub automatically runs lint and tests and builds the Docker images. Green tick = healthy code. | `.github/workflows/ci.yml` |

### The 5 parts that run (the "services")

| Service | What it is | Port |
|---|---|---|
| `db` | PostgreSQL 17 with pgvector: users, documents, chunks, conversations, messages | 5433 on your Mac |
| `redis` | The job queue | 6379 |
| `api` | FastAPI backend: all the logic and endpoints | 8000 (docs at `/docs`) |
| `worker` | Processes uploads in the background | none |
| `ui` | Streamlit web app (what users see) | 8501 |

---

## 4. Where everything lives in the code

| File | What it does |
|---|---|
| `app/main.py` | Starts the app: checks settings, connects the DB, loads the reranker, registers routes |
| `app/config.py` | All settings, read from `.env` (models, chunk size, top-k, prices…) |
| `app/db.py` | Database tables (created automatically on start) and the connection pool |
| `app/auth.py` | Register, login, JWT, "who is the current user" |
| `app/routers/documents.py` | Upload, list, download, reprocess, delete documents |
| `app/routers/chat.py` | The chat endpoint (streaming), conversations, `/stats` |
| `app/ingest.py` | Reading PDFs, OCR fallback, chunking, saving chunks |
| `app/ocr.py` | Tesseract or vision-LLM OCR |
| `app/retrieval.py` | The hybrid search SQL + reranking |
| `app/rerank.py` | The local cross-encoder model |
| `app/llm.py` | Talking to the LLM: embeddings, answers, question rewriting |
| `app/jobs.py`, `app/worker.py` | Putting jobs in the queue and the worker that runs them |
| `app/storage.py` | Saving files on disk or on AWS S3 |
| `app/observability.py` | Optional Langfuse tracing |
| `ui/streamlit_app.py` | The whole web UI |
| `evals/` | Sample documents, 34 test questions, and the scoring script |
| `tests/` | 34 automated tests |

**How to read the code (suggested order):** `app/config.py` → `app/db.py` → `app/ingest.py` → `app/llm.py` →
`app/retrieval.py` → `app/routers/chat.py` → `app/auth.py` → `app/worker.py`. That's the full story in order.

---

## 5. Things you need to set up

| What | Required? | Where to get it | Goes into `.env` as |
|---|---|---|---|
| **Gemini API key** | Yes (or use `LLM_PROVIDER=fake` for testing) | <https://aistudio.google.com/apikey> → sign in with Google → **Create API key** → copy | `LLM_API_KEY` |
| **JWT secret** | Yes | Run `python3 -c "import secrets; print(secrets.token_urlsafe(48))"` and copy the output | `JWT_SECRET` (must be 32+ characters) |
| Langfuse keys | Optional | <https://cloud.langfuse.com> → sign up → create project → Settings → API keys | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` |
| Docker Desktop | Yes | <https://www.docker.com/products/docker-desktop/> | — |
| uv | Only for running without Docker / evals | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | — |
| GitHub account | Yes (to push) | already have: `github.com/vis2712` | — |
| AWS account | For deploying | Section 9 | — |

**About the Gemini free tier:** it's free, but has per-minute and per-day request limits. Fine for a demo and
for evals (the eval runner uses low concurrency). If you see `429` errors, wait a minute.

**Never share or commit `.env`.** It's already in `.gitignore`. If a key ever leaks (e.g. pushed to GitHub),
delete it at the provider and create a new one.

### Create your `.env`

```bash
cd ~/Desktop/documind
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"    # copy this
open -e .env      # opens in TextEdit
```

Change only these two lines and save:

```
LLM_API_KEY=AIza...your-gemini-key...
JWT_SECRET=...the long random string you generated...
```

---

## 6. Run it on your Mac

### Option A: everything in Docker (easiest, use this)

```bash
cd ~/Desktop/documind
docker compose up -d --build        # first time takes ~5 minutes
docker compose ps                   # all should be "Up" (api shows "healthy")
open http://localhost:8501
```

Then in the browser:

1. **Register** with any email + password (8+ characters), then log in.
2. In the sidebar, upload a PDF. Status goes `queued → processing → ready` (auto-refreshes).
   Try the sample files in `evals/corpus/` (create them first with `uv run python -m evals.make_corpus`).
3. Ask a question. You'll see the answer stream in, with sources you can expand.
4. Ask a follow-up like *"and what about managers?"* to see memory working.
5. Open the **Usage** tab to see latency, tokens and cost.

Useful commands:

```bash
docker compose logs -f api worker   # watch what's happening (Ctrl+C to stop watching)
docker compose restart api worker   # after changing .env
docker compose down                 # stop everything (your data is kept)
docker compose down -v              # stop AND delete all data (fresh start)
```

API docs: <http://localhost:8000/docs> → click **Authorize**, log in, and try every endpoint.

### Option B: for coding with live reload

```bash
docker compose up -d db redis
uv sync
uv run fastapi dev app/main.py                 # terminal 1 (API)
uv run python -m app.worker                    # terminal 2 (worker)
uv run streamlit run ui/streamlit_app.py       # terminal 3 (UI)
```

For OCR on your Mac in this mode: `brew install tesseract` (Docker already has it).

---

## 7. Tests and evals

```bash
uv run pytest -q                                  # 34 tests, ~15 seconds, no API key needed
uv run ruff check . && uv run ruff format --check .   # code style
```

**Evals** (needs your real key; takes a few minutes and uses some of your free quota):

```bash
uv run python -m evals.make_corpus     # creates 3 sample PDFs in evals/corpus/
uv run python -m evals.run             # runs 34 questions, writes evals/results/latest.md
```

Open `evals/results/latest.md` and copy the summary table into the README where it says
`<!-- Paste the summary table ... -->`. Real numbers in your README impress interviewers.

What the numbers mean:

- **hit@1 / hit@5**: how often the correct page is the #1 result / in the top 5 results. Higher is better.
- **MRR**: average of 1/rank of the correct page. 1.0 = always first.
- **correctness**: the judge says the answer has the right facts.
- **faithfulness**: the judge says nothing in the answer was made up.
- **refusal accuracy**: on 4 questions with no answer in the docs, did it say "I don't know"?

Try `RERANKER=none uv run python -m evals.run --no-judge` and compare: this shows how much the reranker helps
(a great interview story: "I measured that reranking improved hit@1 from X to Y").

---

## 8. Push to GitHub in clean batches

Your first 7 commits are already on GitHub. These 9 commits add everything else, one feature at a time, so a
reviewer can read the history like a story.

**Step 1: safety check.** Run this and make sure `.env` is **not** in the list:

```bash
cd ~/Desktop/documind
git status
```

**Step 2: commit in batches** (copy one block at a time):

```bash
git add app/auth.py app/config.py app/db.py pyproject.toml uv.lock tests/conftest.py tests/helpers.py tests/test_auth.py
git commit -m "feat(auth): JWT auth with Argon2id hashing, users table and test fixtures"

git add app/storage.py app/ocr.py app/ingest.py tests/test_chunking.py tests/test_ingest.py
git commit -m "feat(ingest): S3/local storage, OCR for scans and images, idempotent re-ingestion"

git add app/rerank.py app/retrieval.py app/llm.py tests/test_storage_and_rerank.py
git commit -m "feat(retrieval): cross-encoder reranking, document filters, follow-up question rewriting"

git add app/jobs.py app/worker.py
git commit -m "feat(queue): Redis/RQ ingestion worker with retries and backoff"

git add app/routers/ app/main.py app/observability.py tests/test_chat.py
git commit -m "feat(api): conversations with memory, usage/cost tracking, /stats, Langfuse tracing"

git add ui/
git commit -m "feat(ui): Streamlit app with login, uploads, streamed chat with sources, usage tab"

git add evals/
git commit -m "feat(evals): sample corpus, 34-question dataset, retrieval metrics and LLM judge"

git add Dockerfile .dockerignore docker-compose.yml .github/ .gitignore
git commit -m "build: Docker images, full compose stack and GitHub Actions CI"

git add README.md docs/ .env.example
git commit -m "docs: README, setup guide, evals and AWS deploy guide"
```

(This guide is in `docs/`, so it's included in the last commit. If you'd rather keep this guide private, run
`git add README.md docs/DEPLOY_AWS.md .env.example` instead for the last one.)

**Step 3: check and push:**

```bash
git status        # should say "nothing to commit, working tree clean"
git log --oneline # you should see 16 commits
git push
```

**Step 4:** open <https://github.com/vis2712/documind> → **Actions** tab. The CI run should go green in a few
minutes. If it's red, click it, copy the error, and send it to me.

**Later changes** (after you edit something):

```bash
git add <files you changed>
git commit -m "fix(chat): short description of what you fixed"
git push
```

Commit message style used here: `feat(...)` new feature, `fix(...)` bug fix, `docs:` documentation,
`build:` Docker/CI, `test:` tests only, `refactor:` code cleanup without behaviour change.

---

## 9. Deploy to AWS (free tier), end to end

**The plan:** one small Linux server on AWS (an **EC2** instance) runs the exact same `docker compose` you run on
your Mac. Simple, cheap, and easy to explain.

```
Your browser ──► http://<server-ip>:8501  ──►  EC2 t3.small (Ubuntu)
                                               └─ docker compose: ui · api · worker · postgres · redis
```

> **If you're also deploying Project 2 (ActionPilot):** both projects can run on this **same server**. Do this
> section first, then follow the "Deploy" section in ActionPilot's guide, which adds it to the same machine.

### How the AWS free tier works now (accounts created after 15 July 2025)

- New accounts get **$100 in credits** at sign-up, plus up to **$100 more** for completing small tasks in the
  console (like launching an EC2 instance or setting a budget).
- On the **Free plan** you can't be charged. It lasts **6 months** or until credits run out, whichever comes
  first. After that you must upgrade to the Paid plan to keep running.
- Free-plan-eligible servers include **t3.micro, t3.small**, t4g.micro, t4g.small and a couple of bigger ones.
- We use **t3.small** (2 vCPU, 2 GB RAM). The micro (1 GB) is too small for the reranker + OCR.
- Rough cost: t3.small ≈ $16/month + disk ≈ $2 + public IP ≈ $3.6 = **≈ $22/month of credits**, so $100–200
  of credits lasts roughly 4–9 months. Both projects on one server = same cost.

Check the current details on <https://aws.amazon.com/free/>; AWS changes these rules from time to time.

### Step 1: Create the AWS account (one time, ~15 min)

1. Go to <https://aws.amazon.com/> → **Create an AWS account**.
2. Email, account name (e.g. `vishal-portfolio`), verify the email.
3. Choose the **Free plan** when asked.
4. Add a card (needed for verification; on the Free plan you won't be charged), verify your phone.
5. Sign in to the **console** as the root user.
6. **Turn on MFA** (very important): top right, your name → **Security credentials** → **Assign MFA device** →
   use Google Authenticator or similar on your phone.

### Step 2: Set a budget alert (2 min, also earns credits)

1. Search bar at the top → **Billing and Cost Management** → **Budgets** → **Create budget**.
2. Choose **Use a template** → **Zero spend budget** (or a monthly cost budget of $5).
3. Enter your email → **Create budget**.

You'll get an email if anything starts costing real money.

### Step 3: Pick a region

Top right of the console, choose **Asia Pacific (Mumbai) ap-south-1**: closest to you, so the app feels faster.
Everything below must be done in this same region.

### Step 4: Launch the server (EC2)

1. Search bar → **EC2** → **Launch instance**.
2. **Name:** `portfolio-server`
3. **Application and OS image:** **Ubuntu** → *Ubuntu Server 24.04 LTS* (64-bit **x86**).
4. **Instance type:** `t3.small`. It should show a "Free tier eligible" label. If it doesn't, check the label
   for t3.micro/t4g.small and tell me what you see.
5. **Key pair (login):** **Create new key pair** → name `portfolio-key` → type RSA → format `.pem` →
   **Create**. A file `portfolio-key.pem` downloads. Keep it safe; you can't download it again.
6. **Network settings** → **Edit**:
   - Auto-assign public IP: **Enable**
   - Create security group, name `portfolio-sg`
   - Rule 1 (already there): SSH, port 22, source **My IP**
   - **Add security group rule**: Custom TCP, port **8501**, source **Anywhere (0.0.0.0/0)** (DocuMind UI)
   - (For Project 2 later: add Custom TCP, port **3000**, Anywhere)
   - Do **not** open 5433, 6379 or 8000. Those stay private inside the server.
7. **Configure storage:** change 8 GiB to **25 GiB**, type **gp3** (Docker images need space).
8. Click **Launch instance** → **View all instances**. Wait until **Instance state = Running** and
   **Status checks = 2/2 passed**.
9. Click the instance and copy the **Public IPv4 address** (e.g. `13.233.10.20`). That's `<server-ip>` below.

### Step 5: Connect to the server

On your Mac:

```bash
mv ~/Downloads/portfolio-key.pem ~/.ssh/
chmod 400 ~/.ssh/portfolio-key.pem          # SSH refuses keys that others can read
ssh -i ~/.ssh/portfolio-key.pem ubuntu@<server-ip>
```

Type `yes` the first time. You're now "inside" the server (the prompt changes to `ubuntu@ip-...`).

(No terminal? EC2 console → select instance → **Connect** → **EC2 Instance Connect** → **Connect** opens a
terminal in the browser.)

### Step 6: Prepare the server (one time)

Copy-paste these on the server:

```bash
# 1. Add 4 GB of swap: extra "backup memory" on disk so the 2 GB server doesn't run out of RAM
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h                                       # "Swap: 4.0Gi" means it worked

# 2. Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu                # lets you run docker without sudo
exit
```

Log in again (so the docker permission applies) and check:

```bash
ssh -i ~/.ssh/portfolio-key.pem ubuntu@<server-ip>
docker --version && docker compose version
```

### Step 7: Get the code onto the server

If your repo is **public** (recommended for a portfolio):

```bash
git clone https://github.com/vis2712/documind.git
cd documind
```

If it's **private**: GitHub → your avatar → **Settings** → **Developer settings** → **Personal access tokens** →
**Fine-grained tokens** → **Generate new token** → only the `documind` repo, permission *Contents: Read-only* →
copy the token. Then `git clone https://github.com/vis2712/documind.git` and paste the token when it asks
for a password.

### Step 8: Create `.env` on the server

```bash
cp .env.example .env
python3 -c "import secrets; print(secrets.token_urlsafe(48))"   # copy this (a NEW secret, not your Mac one)
nano .env
```

In nano: set `LLM_API_KEY=` and `JWT_SECRET=`. Save with **Ctrl+O**, **Enter**, exit with **Ctrl+X**.
(The database/Redis addresses in `.env` don't matter here: docker compose overrides them.)

### Step 9: Start it

```bash
docker compose up -d --build       # first build on the server takes ~10 minutes
docker compose ps                  # api "healthy", the others "Up"
docker compose logs -f api worker  # Ctrl+C to stop watching
```

Open **`http://<server-ip>:8501`** in your browser. Register, upload, ask. 🎉 That's your live demo link.

### Step 10: Updating after you push new code

```bash
ssh -i ~/.ssh/portfolio-key.pem ubuntu@<server-ip>
cd documind
git pull
docker compose up -d --build
```

### Step 11 (optional but nice): a fixed IP and HTTPS

- **Fixed IP:** the public IP changes if you *stop* and *start* the instance. To keep it: EC2 → **Elastic IPs** →
  **Allocate** → select it → **Associate** → choose your instance. (Costs the same as the normal public IP while
  attached; it's charged if you leave it *unattached*, so release it when you delete the server.)
- **HTTPS with a free name:** see "HTTPS for both apps" in ActionPilot's guide, which sets up one Caddy server for
  both projects with addresses like `https://documind.<server-ip>.nip.io`.

### Saving credits

- **Stop** the instance when you're not job-hunting actively (EC2 → Instance state → Stop). You don't pay for
  the server while it's stopped (you still pay ~$2/month for the disk). Start it before interviews.
- To delete everything: **Terminate** the instance, delete the volume if it remains, release any Elastic IP.

### Production version (for interviews)

Read `docs/DEPLOY_AWS.md` "Option 2": ECS Fargate + RDS + ElastiCache + S3 + Secrets Manager + ALB. You don't
need to build it (it costs ~$75/month), but explain it: *"For the demo I run one EC2 box with compose. In
production I'd move Postgres to RDS, Redis to ElastiCache, files to S3 (already supported with
`STORAGE_BACKEND=s3`), and run the API and workers as separate auto-scaling ECS services."*

---

## 10. Interview questions and answers

**Q: Walk me through your RAG pipeline.**
Upload → stored, job queued → worker extracts text per page, OCR for scans → chunks of 1000 chars with 150 overlap
→ Gemini embeddings → Postgres/pgvector. Question → rewrite with chat history → hybrid search (vector + full-text,
merged with RRF) → cross-encoder rerank top 20 to top 5 → grounded prompt with numbered sources → streamed answer
with citations → citation check, usage and cost saved.

**Q: Why hybrid search, not just vectors?**
Embeddings are bad at exact tokens like invoice numbers ("KL-2026-0147"), GSTINs or names. Keyword search is
bad at paraphrases ("vacation" vs "leave"). Hybrid catches both. RRF merges by rank, so I don't need to compare
cosine scores with keyword scores.

**Q: Why a reranker?**
Embedding search compares two separately-made vectors (fast but rough). A cross-encoder reads question and
chunk together (accurate but slow). So: cheap search for 30 candidates, expensive rerank for the top 20, keep 5.

**Q: How do you stop hallucinations?**
Only retrieved sources go in the prompt; the prompt says answer only from them and say "I don't know" otherwise;
every claim needs a citation; I flag citations to sources that weren't given; and I measure faithfulness and
refusal accuracy in evals.

**Q: How do you choose chunk size?**
Started at 1000 characters with 150 overlap and chunks never cross pages (so citations are exact). Then I
tune it with the eval set: change one setting, re-run, compare hit@k.

**Q: How is it multi-tenant?**
JWT identifies the user; every SQL query filters by `owner_id`, including inside the vector search. Other users'
documents return 404, not 403.

**Q: Why Postgres instead of Pinecone/Qdrant?**
One database for app data, vectors and keywords: tenant filters in the same query, transactions for
re-ingestion, one thing to back up. pgvector with HNSW handles millions of chunks. I'd consider a dedicated
vector DB at tens of millions or for special features.

**Q: Why a queue and worker?**
OCR and embedding take seconds to minutes. The API returns 202 immediately; workers scale separately; jobs
survive restarts and retry with backoff (10s, 30s, 90s). Re-running is safe because chunks are replaced in one
transaction.

**Q: How do you know it works / how do you measure quality?**
34-question eval set: hit@1, hit@5 and MRR for retrieval; LLM-judge correctness and faithfulness for answers;
refusal accuracy on unanswerable questions; p50/p95 latency and cost. I compare runs when I change settings.

**Q: What would you improve next?**
Semantic cache (Redis) for repeated questions, PII masking before sending text to the LLM, S3 + RDS in production,
autoscaling workers on queue length, more file types (DOCX, HTML).

**Q: How much does one answer cost?**
Shown per answer in the UI. With Gemini 2.5 Flash, a typical answer (~1,500 input tokens, ~200 output) is a
fraction of a cent. Say the real number from your Usage tab.

---

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `JWT_SECRET must be at least 32 characters` in api logs | Generate one with the `secrets` command and put it in `.env`, then `docker compose up -d` |
| Document stuck on `queued` | Worker not running: `docker compose ps`, then `docker compose logs worker` |
| Document `failed` with "no text… OCR" | Scanned PDF and OCR is off. Use Docker (has Tesseract) or set `OCR_ENGINE=llm` |
| `401` / "Not authenticated" | Token expired (60 min). Log out and in again |
| `429` or "quota" errors | Gemini free tier limit hit. Wait a minute; run evals with `--concurrency 1` |
| `404 model not found` | Check `CHAT_MODEL` / `EMBED_MODEL` names in `.env` against Google AI Studio |
| Changed `EMBED_DIM` and now errors | The DB column size is fixed. `docker compose down -v` (deletes data) and start again |
| Port already in use | Something else uses 8000/8501/5433. Stop it, or change the left number in `ports:` in `docker-compose.yml` |
| Server very slow / build killed | Out of memory: check swap with `free -h` (Step 6) |
| Can't open `http://<ip>:8501` | Security group missing port 8501 rule, or the instance was restarted and the IP changed |
| `Permission denied (publickey)` on SSH | Use `ubuntu@` (not `root@`), the right `.pem` file, and `chmod 400` on it |
