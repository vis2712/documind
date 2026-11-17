"""OCR for scanned PDF pages and image uploads.

Engines:
- tesseract: free, local, fast; good on clean scans. Needs the `tesseract` binary (installed in Docker).
- llm: sends the page image to a vision model; better on photos, tables and handwriting, costs tokens.
- auto: tesseract if installed, otherwise the vision model (unless running in fake mode)."""

import asyncio
import io
import shutil
from functools import cache

from PIL import Image

from app.config import settings
from app.llm import vision_ocr


@cache
def resolve_engine() -> str:
    engine = settings.ocr_engine
    if engine == "auto":
        if shutil.which("tesseract"):
            return "tesseract"
        return "llm" if settings.llm_provider != "fake" else "none"
    return engine


def needs_ocr(text: str) -> bool:
    """A page with (almost) no text layer is most likely a scanned image."""
    return len(text.strip()) < settings.ocr_min_chars


def _prepare(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    longest = max(img.size)
    if longest > 3000:  # keep vision-model requests small and tesseract fast
        img.thumbnail((3000, 3000))
    return img


def _tesseract(img: Image.Image) -> str:
    import pytesseract

    return pytesseract.image_to_string(img)


async def ocr_image(img: Image.Image) -> str:
    engine = resolve_engine()
    if engine == "none":
        return ""
    img = _prepare(img)
    if engine == "tesseract":
        return (await asyncio.to_thread(_tesseract, img)).strip()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return (await vision_ocr(buf.getvalue())).strip()
