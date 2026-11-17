"""File -> pages (text layer or OCR) -> chunks -> embeddings -> pgvector."""

import asyncio
import io
import logging
from dataclasses import dataclass
from uuid import UUID

from PIL import Image
from pypdf import PdfReader

from app.config import settings
from app.db import get_pool
from app.llm import embed
from app.ocr import needs_ocr, ocr_image
from app.storage import get_storage

log = logging.getLogger(__name__)

# Try to split on big boundaries first (paragraphs), then smaller ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]
IMAGE_TYPES = {"image/png", "image/jpeg"}


@dataclass
class Page:
    number: int  # 1-based, so citations match what a PDF viewer shows
    text: str
    is_ocr: bool = False


def _pdf_text_pages(data: bytes) -> list[str]:
    reader = PdfReader(io.BytesIO(data))
    return [(page.extract_text() or "").replace("\x00", "").strip() for page in reader.pages]


def _render_pdf_page(data: bytes, index: int) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        return pdf[index].render(scale=settings.ocr_dpi / 72).to_pil()
    finally:
        pdf.close()


async def extract_pages(data: bytes, content_type: str) -> list[Page]:
    if content_type in IMAGE_TYPES:
        img = await asyncio.to_thread(lambda: Image.open(io.BytesIO(data)))
        return [Page(1, await ocr_image(img), is_ocr=True)]

    texts = await asyncio.to_thread(_pdf_text_pages, data)  # pypdf is sync and CPU-bound
    pages = []
    for i, text in enumerate(texts):
        if needs_ocr(text):
            img = await asyncio.to_thread(_render_pdf_page, data, i)
            ocr_text = await ocr_image(img)
            if len(ocr_text) > len(text):
                pages.append(Page(i + 1, ocr_text, is_ocr=True))
                continue
        pages.append(Page(i + 1, text))
    return pages


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


async def _update(doc_id: UUID, **fields) -> None:
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    async with get_pool().connection() as conn:
        await conn.execute(f"UPDATE documents SET {sets} WHERE id = %(id)s", {"id": doc_id, **fields})


async def ingest_document(doc_id: UUID) -> None:
    """Does the work and raises on failure. Safe to re-run: old chunks are replaced atomically."""
    async with get_pool().connection() as conn:
        cur = await conn.execute("SELECT owner_id, storage_key, content_type FROM documents WHERE id = %s", (doc_id,))
        doc = await cur.fetchone()
    if doc is None:
        log.info("Document %s was deleted before ingestion; skipping", doc_id)
        return

    data = await get_storage().read(doc["storage_key"])
    pages = await extract_pages(data, doc["content_type"])

    records = [
        (page.number, idx, chunk, page.is_ocr)
        for page in pages
        for idx, chunk in enumerate(chunk_text(page.text, settings.chunk_size, settings.chunk_overlap))
    ]
    if not records:
        raise ValueError("No text found. If this is a scan, enable OCR (install tesseract or set OCR_ENGINE=llm).")

    vectors = await embed([content for _, _, content, _ in records])

    async with get_pool().connection() as conn, conn.transaction():
        await conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))  # idempotent retries
        async with conn.cursor() as cur:
            await cur.executemany(
                "INSERT INTO chunks (document_id, owner_id, page, chunk_index, content, embedding, is_ocr) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(doc_id, doc["owner_id"], p, i, c, v, o) for (p, i, c, o), v in zip(records, vectors, strict=True)],
            )
        await conn.execute(
            "UPDATE documents SET status = 'ready', error = NULL, num_pages = %s, num_chunks = %s, ocr_pages = %s "
            "WHERE id = %s",
            (len(pages), len(records), sum(p.is_ocr for p in pages), doc_id),
        )
    log.info(
        "Ingested %s: %d pages (%d OCR), %d chunks", doc_id, len(pages), sum(p.is_ocr for p in pages), len(records)
    )


async def run_ingestion(doc_id: UUID, attempt: int, final_attempt: bool) -> None:
    """Wraps ingest_document with status tracking. Raises on failure so the queue can retry."""
    await _update(doc_id, status="processing", attempts=attempt)
    try:
        await ingest_document(doc_id)
    except Exception as e:
        log.exception("Ingestion of %s failed (attempt %d)", doc_id, attempt)
        if final_attempt:
            await _update(doc_id, status="failed", error=str(e)[:500])
        else:
            await _update(doc_id, status="queued", error=f"Attempt {attempt} failed, retrying: {str(e)[:300]}")
        raise
