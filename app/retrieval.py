"""Hybrid search (vectors + Postgres full-text, fused with Reciprocal Rank Fusion), then reranking."""

import re
from uuid import UUID

from app.config import settings
from app.db import get_pool
from app.llm import embed
from app.observability import traced
from app.rerank import rerank

# RRF: score = sum over result lists of 1 / (60 + rank). Rewards chunks that rank well in BOTH searches,
# without needing to compare cosine distance with ts_rank (they are on different scales).
HYBRID_SQL = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM chunks
    WHERE owner_id = %(owner)s
      AND (%(doc_ids)s::uuid[] IS NULL OR document_id = ANY(%(doc_ids)s::uuid[]))
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(n)s
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('english', %(kwq)s) AS query
    WHERE owner_id = %(owner)s AND tsv @@ query
      AND (%(doc_ids)s::uuid[] IS NULL OR document_id = ANY(%(doc_ids)s::uuid[]))
    ORDER BY ts_rank_cd(tsv, query) DESC
    LIMIT %(n)s
),
fused AS (
    SELECT id, sum(1.0 / (60 + rank)) AS score
    FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) AS both_lists
    GROUP BY id
)
SELECT c.id, c.document_id, d.filename, c.page, c.content, c.is_ocr, f.score
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT %(k)s
"""


def keyword_query(question: str) -> str:
    # "a or b or c": match ANY word (Postgres drops stop words). The default would require ALL words.
    return " or ".join(re.findall(r"\w+", question))


@traced("hybrid-search", as_type="retriever")
async def hybrid_search(
    owner_id: str, question: str, k: int | None = None, document_ids: list[UUID] | None = None
) -> list[dict]:
    [qvec] = await embed([question])
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            HYBRID_SQL,
            {
                "qvec": qvec,
                "owner": owner_id,  # tenant isolation: users only ever search their own chunks
                "kwq": keyword_query(question),
                "doc_ids": document_ids or None,
                "n": settings.candidates,
                "k": k or settings.top_k,
            },
        )
        return await cur.fetchall()


async def retrieve(owner_id: str, question: str, document_ids: list[UUID] | None = None) -> list[dict]:
    """Full pipeline: hybrid search for candidates, then rerank down to top_k."""
    k = settings.rerank_candidates if settings.reranker != "none" else settings.top_k
    candidates = await hybrid_search(owner_id, question, k=k, document_ids=document_ids)
    return await rerank(question, candidates, settings.top_k)
