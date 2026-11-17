"""Where uploaded files live: local disk for development, S3 in production.

The API saves the file and the worker reads it back, so both must point at the same storage
(a shared volume in docker compose, an S3 bucket on AWS)."""

import asyncio
from pathlib import Path
from typing import Protocol

from app.config import settings


class Storage(Protocol):
    async def save(self, key: str, data: bytes, content_type: str) -> None: ...
    async def read(self, key: str) -> bytes: ...
    async def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):  # keys are server-generated, but be safe
            raise ValueError("Invalid storage key")
        return path

    async def save(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_bytes, data)

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self._path(key).read_bytes)

    async def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage:
    def __init__(self, bucket: str, prefix: str):
        import boto3  # credentials come from the environment / ECS task role

        self.bucket, self.prefix = bucket, prefix
        self.s3 = boto3.client("s3")

    async def save(self, key: str, data: bytes, content_type: str) -> None:
        await asyncio.to_thread(
            self.s3.put_object, Bucket=self.bucket, Key=self.prefix + key, Body=data, ContentType=content_type
        )

    async def read(self, key: str) -> bytes:
        obj = await asyncio.to_thread(self.s3.get_object, Bucket=self.bucket, Key=self.prefix + key)
        return await asyncio.to_thread(obj["Body"].read)

    async def delete(self, key: str) -> None:
        await asyncio.to_thread(self.s3.delete_object, Bucket=self.bucket, Key=self.prefix + key)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        if settings.storage_backend == "s3":
            if not settings.s3_bucket:
                raise RuntimeError("STORAGE_BACKEND=s3 needs S3_BUCKET")
            _storage = S3Storage(settings.s3_bucket, settings.s3_prefix)
        else:
            _storage = LocalStorage(settings.upload_dir)
    return _storage
