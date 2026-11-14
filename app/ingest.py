"""PDF -> pages -> chunks -> embeddings -> pgvector."""

import asyncio
import logging
from uuid import UUID

from pypdf import PdfReader

from app.config import settings
from app.db import get_pool
from app.llm import embed

log = logging.getLogger(__name__)

# Try to split on big boundaries first (paragraphs), then smaller ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def extract_pages(path: str) -> list[tuple[int, str]]:
    """Returns (page_number, text) with 1-based page numbers so citations match the PDF viewer."""
    reader = PdfReader(path)
    return [(i + 1, (page.extract_text() or "").replace("\x00", "").strip()) for i, page in enumerate(reader.pages)]


def _atoms(text: str, size: int, seps: list[str]) -> list[str]:
    """Split text into pieces no longer than `size`, using the largest separator that works."""
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    out: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        out.extend(_atoms(piece, size, rest) if len(piece) > size else [piece])
    return out


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Recursive splitter: pack pieces into ~`size` chunks, repeating ~`overlap` chars between neighbours."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for atom in _atoms(text, size, SEPARATORS):
        if current and current_len + len(atom) > size:
            chunks.append("".join(current).strip())
            # Carry the tail of the previous chunk forward so context isn't cut mid-thought.
            tail: list[str] = []
            tail_len = 0
            for a in reversed(current):
                if tail_len + len(a) > overlap or tail_len + len(a) + len(atom) > size:
                    break
                tail.insert(0, a)
                tail_len += len(a)
            current, current_len = tail, tail_len
        current.append(atom)
        current_len += len(atom)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


async def _set_status(doc_id: UUID, status: str, **fields) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in ["status", *fields])
    async with get_pool().connection() as conn:
        await conn.execute(f"UPDATE documents SET {sets} WHERE id = %(id)s", {"id": doc_id, "status": status, **fields})


async def ingest_document(doc_id: UUID, owner_id: str, path: str) -> None:
    """Runs in the background after upload. Never raises: failures are stored on the document."""
    try:
        await _set_status(doc_id, "processing")
        pages = await asyncio.to_thread(extract_pages, path)  # pypdf is sync/CPU-bound

        records = [
            (page, idx, chunk)
            for page, text in pages
            for idx, chunk in enumerate(chunk_text(text, settings.chunk_size, settings.chunk_overlap))
        ]
        if not records:
            raise ValueError("No extractable text. This looks like a scanned PDF (OCR is not added yet).")

        vectors = await embed([content for _, _, content in records])

        async with get_pool().connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    "INSERT INTO chunks (document_id, owner_id, page, chunk_index, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [(doc_id, owner_id, p, i, c, v) for (p, i, c), v in zip(records, vectors)],
                )
        await _set_status(doc_id, "ready", num_pages=len(pages), num_chunks=len(records))
        log.info("Ingested %s: %d pages, %d chunks", doc_id, len(pages), len(records))
    except Exception as e:
        log.exception("Ingestion failed for %s", doc_id)
        await _set_status(doc_id, "failed", error=str(e)[:500])
