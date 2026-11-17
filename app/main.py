import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.auth import router as auth_router
from app.config import settings
from app.db import close_db, get_pool, init_db
from app.jobs import check_queue
from app.observability import ENABLED as TRACING
from app.observability import flush
from app.ocr import resolve_engine
from app.rerank import warm_up
from app.routers.chat import router as chat_router
from app.routers.documents import router as documents_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if len(settings.jwt_secret) < 32:
        raise RuntimeError("JWT_SECRET must be set in .env (32+ chars). See .env.example.")
    await init_db()
    await asyncio.to_thread(check_queue)
    await asyncio.to_thread(warm_up)
    log.info(
        "DocuMind ready: llm=%s chat_model=%s queue=%s storage=%s ocr=%s reranker=%s tracing=%s",
        settings.llm_provider,
        settings.chat_model,
        settings.queue_backend,
        settings.storage_backend,
        resolve_engine(),
        settings.reranker,
        TRACING,
    )
    yield
    await close_db()
    await asyncio.to_thread(flush)  # send any buffered traces before exiting


app = FastAPI(title="DocuMind", description="Chat with your documents, with citations", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health", tags=["ops"])
async def health():
    async with get_pool().connection() as conn:
        await conn.execute("SELECT 1")
    return {"status": "ok"}
