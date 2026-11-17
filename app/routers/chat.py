import json
import logging
import time
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from app.auth import current_user
from app.config import settings
from app.db import get_pool
from app.llm import NO_ANSWER, Answer, cited_tags, condense_question, stream_answer
from app.observability import traced
from app.retrieval import retrieve

log = logging.getLogger("app.chat")
router = APIRouter(tags=["chat"])
User = Annotated[str, Depends(current_user)]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: UUID | None = None  # omit to start a new conversation
    document_ids: list[UUID] | None = Field(default=None, max_length=50)  # limit search to these documents


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


async def _conversation(user: str, conversation_id: UUID | None, question: str) -> UUID:
    async with get_pool().connection() as conn:
        if conversation_id is None:
            cur = await conn.execute(
                "INSERT INTO conversations (owner_id, title) VALUES (%s, %s) RETURNING id", (user, question[:80])
            )
        else:
            cur = await conn.execute(
                "SELECT id FROM conversations WHERE id = %s AND owner_id = %s", (conversation_id, user)
            )
        row = await cur.fetchone()
    if row is None:
        raise HTTPException(404, "Conversation not found")
    return row["id"]


async def _history(conversation_id: UUID) -> list[dict]:
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT role, content FROM (SELECT id, role, content FROM messages WHERE conversation_id = %s "
            "ORDER BY id DESC LIMIT %s) recent ORDER BY id",
            (conversation_id, settings.history_turns),
        )
        return await cur.fetchall()


async def _save_turn(conversation_id: UUID, question: str, search_query: str, answer: str, sources, usage) -> int:
    async with get_pool().connection() as conn, conn.transaction():
        await conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s)", (conversation_id, question)
        )
        cur = await conn.execute(
            "INSERT INTO messages (conversation_id, role, content, search_query, sources, usage) "
            "VALUES (%s, 'assistant', %s, %s, %s, %s) RETURNING id",
            (conversation_id, answer, search_query, Jsonb(sources), Jsonb(usage)),
        )
        await conn.execute("UPDATE conversations SET updated_at = now() WHERE id = %s", (conversation_id,))
        return (await cur.fetchone())["id"]


@traced("chat")
async def _chat_events(user: str, req: ChatRequest, conversation_id: UUID):
    started = time.perf_counter()
    answer = Answer()
    sources: list[dict] = []
    search_query = req.question
    usage: dict = {}
    try:
        history = await _history(conversation_id)
        search_query = await condense_question(history, req.question)
        yield sse("meta", {"conversation_id": conversation_id, "search_query": search_query})

        t0 = time.perf_counter()
        chunks = await retrieve(user, search_query, req.document_ids)
        retrieval_ms = round((time.perf_counter() - t0) * 1000)
        sources = [
            {
                "tag": f"S{i}",
                "document_id": str(c["document_id"]),
                "filename": c["filename"],
                "page": c["page"],
                "is_ocr": c["is_ocr"],
                "score": round(float(c.get("rerank_score", c["score"])), 4),
                "content": c["content"],
            }
            for i, c in enumerate(chunks, 1)
        ]
        yield sse("sources", sources)

        first_token_ms = None
        if not chunks:
            answer.text = NO_ANSWER
            yield sse("token", NO_ANSWER)
        else:
            async for token in stream_answer(req.question, chunks, history, answer):
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                yield sse("token", token)

        usage = {
            "prompt_tokens": answer.usage.prompt_tokens,
            "completion_tokens": answer.usage.completion_tokens,
            "cost_usd": round(answer.usage.cost_usd, 6),
            "retrieval_ms": retrieval_ms,
            "first_token_ms": first_token_ms,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "model": settings.chat_model if settings.llm_provider != "fake" else "fake",
        }
        # Guardrail: flag citations to sources that don't exist (a sign of hallucination).
        invalid = sorted(cited_tags(answer.text) - {s["tag"] for s in sources})
        message_id = await _save_turn(conversation_id, req.question, search_query, answer.text, sources, usage)
        log.info(
            "chat user=%s conv=%s latency=%sms tokens=%s/%s cost=$%.5f sources=%d invalid_citations=%s",
            user,
            conversation_id,
            usage["latency_ms"],
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["cost_usd"],
            len(sources),
            invalid,
        )
        yield sse("done", {"message_id": message_id, "usage": usage, "invalid_citations": invalid})
    except Exception as e:
        log.exception("Chat failed")
        yield sse("error", {"message": f"{type(e).__name__}: {str(e)[:300]}"})


@router.post("/chat")
async def chat(req: ChatRequest, user: User):
    """Streams Server-Sent Events: `meta`, `sources`, many `token`s, then `done` (or `error`)."""
    conversation_id = await _conversation(user, req.conversation_id, req.question)
    return StreamingResponse(
        _chat_events(user, req, conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},  # don't let proxies buffer the stream
    )


@router.get("/conversations")
async def list_conversations(user: User):
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE owner_id = %s "
            "ORDER BY updated_at DESC LIMIT 100",
            (user,),
        )
        return await cur.fetchall()


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: UUID, user: User):
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = %s AND owner_id = %s",
            (conversation_id, user),
        )
        conversation = await cur.fetchone()
        if conversation is None:
            raise HTTPException(404, "Conversation not found")
        cur = await conn.execute(
            "SELECT id, role, content, search_query, sources, usage, created_at FROM messages "
            "WHERE conversation_id = %s ORDER BY id",
            (conversation_id,),
        )
        conversation["messages"] = await cur.fetchall()
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: UUID, user: User):
    async with get_pool().connection() as conn:
        cur = await conn.execute("DELETE FROM conversations WHERE id = %s AND owner_id = %s", (conversation_id, user))
    if cur.rowcount == 0:
        raise HTTPException(404, "Conversation not found")


@router.get("/stats")
async def stats(user: User):
    """Per-user usage dashboard: documents, questions, tokens, cost and latency."""
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "SELECT count(*) AS documents, count(*) FILTER (WHERE status = 'ready') AS ready, "
            "coalesce(sum(num_pages), 0) AS pages, coalesce(sum(num_chunks), 0) AS chunks "
            "FROM documents WHERE owner_id = %s",
            (user,),
        )
        docs = await cur.fetchone()
        cur = await conn.execute(
            """
            SELECT count(*) AS answers,
                   coalesce(sum((m.usage->>'prompt_tokens')::int), 0) AS prompt_tokens,
                   coalesce(sum((m.usage->>'completion_tokens')::int), 0) AS completion_tokens,
                   coalesce(sum((m.usage->>'cost_usd')::float), 0) AS cost_usd,
                   coalesce(avg((m.usage->>'latency_ms')::float), 0) AS avg_latency_ms,
                   coalesce(percentile_cont(0.95) WITHIN GROUP (ORDER BY (m.usage->>'latency_ms')::float), 0)
                       AS p95_latency_ms
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.owner_id = %s AND m.role = 'assistant' AND m.usage ? 'latency_ms'
            """,
            (user,),
        )
        answers = await cur.fetchone()
    return {"documents": docs, "answers": answers}
