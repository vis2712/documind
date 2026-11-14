import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.db import close_db, get_pool, init_db
from app.ingest import ingest_document
from app.llm import NO_ANSWER, stream_answer
from app.retrieval import hybrid_search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    await init_db()
    yield
    await close_db()


app = FastAPI(title="DocuMind", description="Chat with your documents", lifespan=lifespan)


def current_user(x_user_id: str = Header(min_length=1, max_length=64)) -> str:
    """TEMPORARY auth: trust an X-User-Id header. Replace with JWT verification (next milestone)."""
    return x_user_id


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


DOC_COLUMNS = "id, filename, status, error, num_pages, num_chunks, created_at"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/documents", status_code=202)
async def upload_document(file: UploadFile, background: BackgroundTasks, user: str = Depends(current_user)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported for now")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")

    async with get_pool().connection() as conn:
        cur = await conn.execute(
            f"INSERT INTO documents (owner_id, filename) VALUES (%s, %s) RETURNING {DOC_COLUMNS}",
            (user, file.filename),
        )
        doc = await cur.fetchone()

    path = Path(settings.upload_dir) / f"{doc['id']}.pdf"
    path.write_bytes(data)
    # Returns 202 immediately; ingestion continues after the response is sent.
    background.add_task(ingest_document, doc["id"], user, str(path))
    return doc


@app.get("/documents")
async def list_documents(user: str = Depends(current_user)):
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {DOC_COLUMNS} FROM documents WHERE owner_id = %s ORDER BY created_at DESC", (user,)
        )
        return await cur.fetchall()


@app.get("/documents/{doc_id}")
async def get_document(doc_id: UUID, user: str = Depends(current_user)):
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {DOC_COLUMNS} FROM documents WHERE id = %s AND owner_id = %s", (doc_id, user)
        )
        doc = await cur.fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")  # 404 (not 403) so other users' IDs aren't revealed
    return doc


@app.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: UUID, user: str = Depends(current_user)):
    async with get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM documents WHERE id = %s AND owner_id = %s", (doc_id, user))
    if cur.rowcount == 0:
        raise HTTPException(404, "Document not found")
    (Path(settings.upload_dir) / f"{doc_id}.pdf").unlink(missing_ok=True)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, user: str = Depends(current_user)):
    """Streams Server-Sent Events: one `sources` event, many `token` events, then `done`."""
    chunks = await hybrid_search(user, req.question)

    async def events():
        sources = [
            {"tag": f"S{i}", "document_id": c["document_id"], "filename": c["filename"],
             "page": c["page"], "score": round(float(c["score"]), 4), "preview": c["content"][:200]}
            for i, c in enumerate(chunks, 1)
        ]
        yield sse("sources", sources)
        if not chunks:
            yield sse("token", NO_ANSWER)
        else:
            try:
                async for token in stream_answer(req.question, chunks):
                    yield sse("token", token)
            except Exception as e:
                logging.exception("LLM call failed")
                yield sse("error", str(e)[:300])
        yield sse("done", {})

    return StreamingResponse(events(), media_type="text/event-stream")
