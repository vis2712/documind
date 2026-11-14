import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    text NOT NULL,
    filename    text NOT NULL,
    status      text NOT NULL DEFAULT 'pending',  -- pending | processing | ready | failed
    error       text,
    num_pages   int,
    num_chunks  int,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    owner_id     text NOT NULL,              -- copied here so every search filters by tenant cheaply
    page         int  NOT NULL,
    chunk_index  int  NOT NULL,
    content      text NOT NULL,
    embedding    vector({dim}) NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_owner_idx ON chunks (owner_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
"""

pool: AsyncConnectionPool | None = None


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)  # lets us pass numpy arrays as `vector` params
    conn.row_factory = dict_row


async def init_db() -> None:
    global pool
    # The vector extension must exist before the pool registers the vector type.
    async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        await conn.execute(SCHEMA.format(dim=settings.embed_dim))
    pool = AsyncConnectionPool(settings.database_url, configure=_configure, open=False)
    await pool.open()


async def close_db() -> None:
    if pool:
        await pool.close()


def get_pool() -> AsyncConnectionPool:
    assert pool is not None, "init_db() was not called"
    return pool
