from uuid import UUID

import pytest

import app.ingest
from app.ingest import run_ingestion
from app.ocr import needs_ocr
from tests.helpers import make_pdf, make_png, make_scanned_pdf, new_user, upload


def test_needs_ocr():
    assert needs_ocr("")
    assert needs_ocr("  \n 12 ")
    assert not needs_ocr("This page has a real text layer with plenty of words.")


def test_rejects_unsupported_files(client):
    headers = new_user(client)
    r = client.post("/documents", headers=headers, files={"file": ("notes.pdf", b"just text", "application/pdf")})
    assert r.status_code == 415  # the name says .pdf but the bytes don't


def test_scanned_pdf_page_is_ocrd(client, monkeypatch):
    async def fake_ocr(img):
        return "Office closed on 2 October 2026 for Gandhi Jayanti."

    monkeypatch.setattr(app.ingest, "ocr_image", fake_ocr)
    headers = new_user(client)
    doc = upload(client, headers, make_scanned_pdf(), "scan.pdf")
    assert doc["status"] == "ready", doc
    assert doc["ocr_pages"] == 1


def test_image_upload_is_ocrd(client, monkeypatch):
    async def fake_ocr(img):
        return "Receipt total 450 rupees"

    monkeypatch.setattr(app.ingest, "ocr_image", fake_ocr)
    headers = new_user(client)
    doc = upload(client, headers, make_png(), "receipt.png")
    assert doc["content_type"] == "image/png"
    assert doc["status"] == "ready" and doc["ocr_pages"] == 1


def test_scan_without_ocr_fails_with_a_clear_message(client):
    headers = new_user(client)
    doc = upload(client, headers, make_scanned_pdf(), "scan.pdf")
    assert doc["status"] == "failed"
    assert "OCR" in doc["error"]


def test_failed_attempt_is_requeued_then_succeeds_without_duplicates(client, monkeypatch):
    headers = new_user(client)
    doc = upload(client, headers, make_pdf(["Refund policy: refunds are processed in 7 days."]), "refund.pdf")
    assert doc["status"] == "ready" and doc["num_chunks"] == 1
    doc_id = UUID(doc["id"])

    real_embed = app.ingest.embed

    async def flaky_embed(texts):
        raise TimeoutError("embedding API timed out")

    # Attempt 1 of 2 fails: the document goes back to "queued" with the error, ready for a retry.
    monkeypatch.setattr(app.ingest, "embed", flaky_embed)
    with pytest.raises(TimeoutError):
        client.portal.call(run_ingestion, doc_id, 1, False)
    doc = client.get(f"/documents/{doc_id}", headers=headers).json()
    assert doc["status"] == "queued" and "retrying" in doc["error"] and doc["attempts"] == 1

    # Attempt 2 succeeds, and re-running replaced the chunks instead of duplicating them.
    monkeypatch.setattr(app.ingest, "embed", real_embed)
    client.portal.call(run_ingestion, doc_id, 2, True)
    doc = client.get(f"/documents/{doc_id}", headers=headers).json()
    assert doc["status"] == "ready" and doc["error"] is None and doc["num_chunks"] == 1


def test_final_failed_attempt_marks_document_failed(client, monkeypatch):
    headers = new_user(client)
    doc = upload(client, headers, make_pdf(["Some content that is long enough to embed."]), "a.pdf")

    async def broken_embed(texts):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(app.ingest, "embed", broken_embed)
    with pytest.raises(RuntimeError):
        client.portal.call(run_ingestion, UUID(doc["id"]), 3, True)
    doc = client.get(f"/documents/{doc['id']}", headers=headers).json()
    assert doc["status"] == "failed" and "quota exceeded" in doc["error"]


def test_download_reprocess_and_delete(client):
    headers = new_user(client)
    pdf = make_pdf(["Reprocess me, this is the content of the page."])
    doc = upload(client, headers, pdf, "r.pdf")

    r = client.get(f"/documents/{doc['id']}/file", headers=headers)
    assert r.status_code == 200 and r.content == pdf and r.headers["content-type"] == "application/pdf"

    assert client.post(f"/documents/{doc['id']}/reprocess", headers=headers).status_code == 202
    assert client.get(f"/documents/{doc['id']}", headers=headers).json()["status"] == "ready"

    assert client.delete(f"/documents/{doc['id']}", headers=headers).status_code == 204
    assert client.get(f"/documents/{doc['id']}/file", headers=headers).status_code == 404
