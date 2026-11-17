"""Optional Langfuse tracing. Everything here is a no-op unless LANGFUSE_* keys are set.

When enabled you get one trace per chat request with nested spans for query rewriting, retrieval,
reranking and the LLM generation (tokens, cost and latency), viewable in the Langfuse UI."""

import os
from collections.abc import Callable

from app.config import settings

ENABLED = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if ENABLED:
    # The Langfuse SDK reads its config from os.environ, but our keys may only be in .env.
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_base_url)


def traced(name: str, as_type: str = "span") -> Callable:
    """Decorator: a Langfuse span when tracing is on, otherwise returns the function unchanged."""
    if not ENABLED:
        return lambda fn: fn
    from langfuse import observe

    return observe(name=name, as_type=as_type)


def flush() -> None:
    if ENABLED:
        from langfuse import get_client

        get_client().flush()
