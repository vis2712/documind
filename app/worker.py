"""Ingestion worker. Run with:  uv run python -m app.worker"""

import asyncio
import logging
import sys
from uuid import UUID

from app import db
from app.config import settings
from app.ingest import run_ingestion
from app.observability import flush

log = logging.getLogger("app.worker")


async def _run(doc_id: UUID, attempt: int, final_attempt: bool) -> None:
    await db.open_pool()  # each job gets its own event loop, so its own pool
    try:
        await run_ingestion(doc_id, attempt, final_attempt)
    finally:
        await db.close_db()
        await asyncio.to_thread(flush)


def ingest_job(doc_id: str) -> None:
    """Called by RQ. Raising makes RQ schedule a retry (if any are left)."""
    from rq import get_current_job

    job = get_current_job()
    retries_left = (job.retries_left or 0) if job else 0
    total = settings.ingest_max_retries + 1
    attempt = total - retries_left
    asyncio.run(_run(UUID(doc_id), attempt=attempt, final_attempt=retries_left == 0))


def main() -> None:
    from redis import Redis
    from rq import SimpleWorker, Worker

    from app.jobs import QUEUE_NAME

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(db.create_schema())
    # macOS doesn't fork safely with some native libraries, so run jobs in-process there.
    worker_class = SimpleWorker if sys.platform == "darwin" else Worker
    worker = worker_class([QUEUE_NAME], connection=Redis.from_url(settings.redis_url))
    log.info("Worker listening on queue %r (%s)", QUEUE_NAME, worker_class.__name__)
    worker.work(with_scheduler=True)  # the scheduler is what runs delayed retries


if __name__ == "__main__":
    main()
