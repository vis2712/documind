import io
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response, UploadFile, status
from PIL import Image

from app.auth import current_user
from app.config import settings
from app.db import get_pool
from app.jobs import enqueue_ingestion
from app.storage import get_storage

router = APIRouter(prefix="/documents", tags=["documents"])
User = Annotated[str, Depends(current_user)]

DOC_COLUMNS = "id, filename, content_type, status, error, num_pages, num_chunks, ocr_pages, attempts, created_at"
EXTENSIONS = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg"}


def detect_type(data: bytes) -> str:
    """Trust the file's bytes, not its name or the client's Content-Type header."""
    if data.startswith(b"%PDF-"):
        return "application/pdf"
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
            fmt = img.format
    except Exception:
        fmt = None
    if fmt == "PNG":
        return "image/png"
    if fmt == "JPEG":
        return "image/jpeg"
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF, PNG and JPEG files are supported")


async def _get_owned(doc_id: UUID, user: str) -> dict:
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {DOC_COLUMNS}, storage_key FROM documents WHERE id = %s AND owner_id = %s", (doc_id, user)
        )
        doc = await cur.fetchone()
    if not doc:
        raise HTTPException(404, "Document not found")  # 404 (not 403) so other users' IDs aren't revealed
    return doc


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile, background: BackgroundTasks, user: User):
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File larger than {settings.max_upload_mb} MB")
    if not data:
        raise HTTPException(400, "Empty file")
    content_type = detect_type(data)

    async with get_pool().connection() as conn:
        cur = await conn.execute(
            "INSERT INTO documents (owner_id, filename, content_type, status) VALUES (%s, %s, %s, 'queued') "
            "RETURNING id",
            (user, (file.filename or "upload")[:255], content_type),
        )
        doc_id = (await cur.fetchone())["id"]
        key = f"{user}/{doc_id}.{EXTENSIONS[content_type]}"
        await get_storage().save(key, data, content_type)
        cur = await conn.execute(
            f"UPDATE documents SET storage_key = %s WHERE id = %s RETURNING {DOC_COLUMNS}", (key, doc_id)
        )
        doc = await cur.fetchone()

    await enqueue_ingestion(doc_id, background)
    return doc


@router.get("")
async def list_documents(user: User):
    async with get_pool().connection() as conn:
        cur = await conn.execute(
            f"SELECT {DOC_COLUMNS} FROM documents WHERE owner_id = %s ORDER BY created_at DESC", (user,)
        )
        return await cur.fetchall()


@router.get("/{doc_id}")
async def get_document(doc_id: UUID, user: User):
    doc = await _get_owned(doc_id, user)
    doc.pop("storage_key")
    return doc


@router.get("/{doc_id}/file")
async def download_file(doc_id: UUID, user: User):
    doc = await _get_owned(doc_id, user)
    data = await get_storage().read(doc["storage_key"])
    safe_name = doc["filename"].replace('"', "")
    return Response(
        data, media_type=doc["content_type"], headers={"Content-Disposition": f'inline; filename="{safe_name}"'}
    )


@router.post("/{doc_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(doc_id: UUID, background: BackgroundTasks, user: User):
    """Re-run ingestion, e.g. after a failure or after changing chunking/OCR settings."""
    await _get_owned(doc_id, user)
    async with get_pool().connection() as conn:
        await conn.execute("UPDATE documents SET status = 'queued', error = NULL WHERE id = %s", (doc_id,))
    await enqueue_ingestion(doc_id, background)
    return {"id": doc_id, "status": "queued"}


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(doc_id: UUID, user: User):
    doc = await _get_owned(doc_id, user)
    async with get_pool().connection() as conn:
        await conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))  # chunks cascade
    if doc["storage_key"]:
        await get_storage().delete(doc["storage_key"])
