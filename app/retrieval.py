"""Hybrid search: vector similarity + Postgres full-text (BM25-style), fused with Reciprocal Rank Fusion."""

import re

from app.config import settings
from app.db import get_pool
from app.llm import embed

# RRF: score = sum over result lists of 1 / (60 + rank). Rewards chunks that rank well in BOTH searches,
# without needing to compare cosine distance with ts_rank (they are on different scales).
HYBRID_SQL = """
WITH vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM chunks
    WHERE owner_id = %(owner)s
    ORDER BY embedding <=> %(qvec)s
    LIMIT %(n)s
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query) DESC) AS rank
    FROM chunks, websearch_to_tsquery('english', %(kwq)s) AS query
    WHERE owner_id = %(owner)s AND tsv @@ query
    ORDER BY ts_rank_cd(tsv, query) DESC
    LIMIT %(n)s
),
fused AS (
    SELECT id, sum(1.0 / (60 + rank)) AS score
    FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) AS both_lists
    GROUP BY id
)
SELECT c.id, c.document_id, d.filename, c.page, c.content, f.score
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC
LIMIT %(k)s
"""


def keyword_query(question: str) -> str:
    # "a or b or c": match ANY word (Postgres drops stop words). The default would require ALL words.
    return " or ".join(re.findall(r"\w+", question))


async def hybrid_search(owner_id: str, question: str, k: int | None = None) -> list[dict]:
    [qvec] = await embed([question])
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            HYBRID_SQL,
            {
                "qvec": qvec,
                "owner": owner_id,  # tenant isolation: users only ever search their own chunks
                "kwq": keyword_query(question),
                "n": settings.candidates,
                "k": k or settings.top_k,
            },
        )
        return await cur.fetchall()
