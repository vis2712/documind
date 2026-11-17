import io
import json
import uuid

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

PASSWORD = "correct-horse-battery"


def make_pdf(pages: list[str]) -> bytes:
    """A real PDF with a text layer: one string per page (newlines become lines)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 800
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 16
        c.showPage()
    c.save()
    return buf.getvalue()


def make_scanned_pdf() -> bytes:
    """A PDF page that contains only an image (no text layer), like a scan."""
    from PIL import Image

    img = Image.new("RGB", (400, 200), "white")
    buf = io.BytesIO()
    img.save(buf, format="PDF")
    return buf.getvalue()


def make_png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (100, 50), "white").save(buf, format="PNG")
    return buf.getvalue()


def new_user(client) -> dict:
    email = f"user-{uuid.uuid4().hex[:10]}@example.com"
    assert client.post("/auth/register", json={"email": email, "password": PASSWORD}).status_code == 201
    token = client.post("/auth/login", data={"username": email, "password": PASSWORD}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def upload(client, headers, data: bytes, filename: str = "doc.pdf") -> dict:
    r = client.post("/documents", headers=headers, files={"file": (filename, data, "application/octet-stream")})
    assert r.status_code == 202, r.text
    # TestClient runs background tasks before returning, so ingestion has finished by now.
    return client.get(f"/documents/{r.json()['id']}", headers=headers).json()


def parse_sse(text: str) -> list[tuple[str, object]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def chat(client, headers, question: str, **body) -> dict[str, object]:
    r = client.post("/chat", headers=headers, json={"question": question, **body})
    assert r.status_code == 200, r.text
    events = parse_sse(r.text)
    out: dict[str, object] = {"tokens": ""}
    for name, data in events:
        if name == "token":
            out["tokens"] += data
        else:
            out[name] = data
    return out
