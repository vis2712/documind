import asyncio

import boto3
import pytest
from moto import mock_aws

from app.storage import LocalStorage, S3Storage


def test_local_storage_roundtrip(tmp_path):
    store = LocalStorage(str(tmp_path))
    asyncio.run(store.save("u1/a.pdf", b"data", "application/pdf"))
    assert asyncio.run(store.read("u1/a.pdf")) == b"data"
    asyncio.run(store.delete("u1/a.pdf"))
    with pytest.raises(FileNotFoundError):
        asyncio.run(store.read("u1/a.pdf"))


def test_local_storage_blocks_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        asyncio.run(LocalStorage(str(tmp_path)).read("../../etc/passwd"))


@mock_aws
def test_s3_storage_roundtrip(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
    boto3.client("s3").create_bucket(Bucket="docs", CreateBucketConfiguration={"LocationConstraint": "ap-south-1"})
    store = S3Storage("docs", "uploads/")
    asyncio.run(store.save("u1/a.pdf", b"data", "application/pdf"))
    assert asyncio.run(store.read("u1/a.pdf")) == b"data"
    assert boto3.client("s3").head_object(Bucket="docs", Key="uploads/u1/a.pdf")["ContentType"] == "application/pdf"


def test_local_reranker_puts_the_relevant_chunk_first(monkeypatch):
    """Downloads a small (~80 MB) ONNX model on first run."""
    import app.rerank as rr

    monkeypatch.setattr(rr.settings, "reranker", "local")
    chunks = [
        {"content": "The office cafeteria serves lunch from 12 to 2."},
        {"content": "Employees receive 24 days of paid annual leave every year."},
        {"content": "Invoices are payable within 30 days."},
    ]
    ranked = asyncio.run(rr.rerank("How much vacation do I get?", chunks, top_k=2))
    assert len(ranked) == 2
    assert "annual leave" in ranked[0]["content"]
