"""Second-stage ranking with a cross-encoder.

Hybrid search is fast because the question and each chunk are embedded separately. A cross-encoder
reads the question and the chunk *together*, so it judges relevance much better, but it is too slow
to run over every chunk. So: retrieve ~20 cheaply, rerank those, keep the best 5.
Runs locally on CPU via ONNX (fastembed), so there is no API cost and no PyTorch install."""

import asyncio
from functools import cache

from app.config import settings
from app.observability import traced


@cache
def _model():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=settings.rerank_model, cache_dir=settings.model_cache_dir)


def warm_up() -> None:
    """Loads (and on first run downloads, ~80 MB) the model so the first question isn't slow."""
    if settings.reranker == "local":
        list(_model().rerank("warm up", ["warm up"]))


@traced("rerank")
async def rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    if settings.reranker == "none" or len(chunks) <= 1:
        return chunks[:top_k]
    texts = [c["content"] for c in chunks]
    scores = await asyncio.to_thread(lambda: list(_model().rerank(question, texts)))
    for chunk, score in zip(chunks, scores, strict=True):
        chunk["rerank_score"] = float(score)
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_k]
