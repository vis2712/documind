"""Hands ingestion off to a background worker.

- rq (default): the job goes into Redis and a separate worker process (`python -m app.worker`) runs it,
  with retries and backoff. The API stays fast, and jobs survive an API restart.
- background: runs inside the API process after the response (no Redis). Used in tests."""

import asyncio
import logging
from uuid import UUID

from fastapi import BackgroundTasks

from app.config import settings
from app.ingest import run_ingestion

log = logging.getLogger(__name__)

QUEUE_NAME = "ingest"
RETRY_INTERVALS = [10, 30, 90]  # seconds between attempts


def _queue():
    from redis import Redis
    from rq import Queue

    return Queue(QUEUE_NAME, connection=Redis.from_url(settings.redis_url))


async def _run_inline(doc_id: UUID) -> None:
    try:
        await run_ingestion(doc_id, attempt=1, final_attempt=True)
    except Exception:
        pass  # already logged and stored on the document


async def enqueue_ingestion(doc_id: UUID, background: BackgroundTasks) -> None:
    if settings.queue_backend == "background":
        background.add_task(_run_inline, doc_id)
        return
    from rq import Retry

    retries = settings.ingest_max_retries
    await asyncio.to_thread(
        _queue().enqueue,
        "app.worker.ingest_job",
        str(doc_id),
        retry=Retry(max=retries, interval=RETRY_INTERVALS[:retries]) if retries else None,
        job_timeout=900,
        description=f"ingest {doc_id}",
    )


def check_queue() -> None:
    """Fail fast at startup if Redis is unreachable."""
    if settings.queue_backend == "rq":
        _queue().connection.ping()
