import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.config import settings

# Idempotent schema: CREATE ... IF NOT EXISTS plus ADD COLUMN IF NOT EXISTS for columns added later,
# so an existing database upgrades itself on startup. (A bigger project would use Alembic.)
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email          text NOT NULL UNIQUE,  -- stored lowercased
    password_hash  text NOT NULL,         -- Argon2id, never the password itself
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    text NOT NULL,
    filename    text NOT NULL,
    status      text NOT NULL DEFAULT 'pending',  -- pending | queued | processing | ready | failed
    error       text,
    num_pages   int,
    num_chunks  int,
    created_at  timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_type text NOT NULL DEFAULT 'application/pdf';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS storage_key  text;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ocr_pages    int;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS attempts     int NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS documents_owner_idx ON documents (owner_id, created_at DESC);

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
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS is_ocr boolean NOT NULL DEFAULT false;
CREATE INDEX IF NOT EXISTS chunks_owner_idx ON chunks (owner_id);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id    text NOT NULL,
    title       text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS conversations_owner_idx ON conversations (owner_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id               bigserial PRIMARY KEY,
    conversation_id  uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             text NOT NULL CHECK (role IN ('user', 'assistant')),
    content          text NOT NULL,
    search_query     text,    -- the standalone question actually used for retrieval
    sources          jsonb,   -- citations shown with an assistant answer
    usage            jsonb,   -- tokens, cost, latency for an assistant answer
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);
"""

pool: AsyncConnectionPool | None = None


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)  # lets us pass numpy arrays as `vector` params
    conn.row_factory = dict_row


async def create_schema() -> None:
    # The vector extension must exist before a pool registers the vector type.
    async with await psycopg.AsyncConnection.connect(settings.database_url, autocommit=True) as conn:
        await conn.execute(SCHEMA.format(dim=settings.embed_dim))


async def open_pool() -> None:
    global pool
    pool = AsyncConnectionPool(settings.database_url, configure=_configure, open=False)
    await pool.open()


async def init_db() -> None:
    await create_schema()
    await open_pool()


async def close_db() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


def get_pool() -> AsyncConnectionPool:
    assert pool is not None, "init_db() was not called"
    return pool
