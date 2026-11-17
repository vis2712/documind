"""Test setup. Env vars are set here, before any `app` module is imported, so settings pick them up."""

import os
import tempfile

import psycopg
import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL", "postgresql://documind:documind@localhost:5433/documind")
TEST_DB = "documind_test"

os.environ.update(
    LLM_PROVIDER="fake",  # no API calls or costs in tests
    JWT_SECRET="test-secret-that-is-long-enough-for-hs256-0123456789",
    DATABASE_URL=f"{ADMIN_URL.rsplit('/', 1)[0]}/{TEST_DB}",  # separate DB: tests never touch your data
    UPLOAD_DIR=tempfile.mkdtemp(prefix="documind-test-"),
    STORAGE_BACKEND="local",
    QUEUE_BACKEND="background",  # ingest in-process; no Redis needed for tests
    RERANKER="none",  # no model download in tests (the rerank test opts in explicitly)
    OCR_ENGINE="none",
    LANGFUSE_PUBLIC_KEY="",
    LANGFUSE_SECRET_KEY="",
)


@pytest.fixture(scope="session")
def client():
    """A TestClient against a fresh `documind_test` database. Skips if Postgres isn't running."""
    try:
        with psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3) as conn:
            conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
            conn.execute(f"CREATE DATABASE {TEST_DB}")
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")

    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:  # `with` runs the lifespan (creates the schema)
        yield c
