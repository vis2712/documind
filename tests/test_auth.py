import io
import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pypdf import PdfWriter

from app.config import settings

PASSWORD = "correct-horse-battery"


def new_email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


def register(client, email: str, password: str = PASSWORD):
    return client.post("/auth/register", json={"email": email, "password": password})


def login(client, email: str, password: str = PASSWORD):
    return client.post("/auth/login", data={"username": email, "password": password})


def auth_headers(client) -> dict:
    email = new_email()
    assert register(client, email).status_code == 201
    token = login(client, email).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_register_login_me(client):
    email = new_email()
    r = register(client, email)
    assert r.status_code == 201
    assert r.json()["email"] == email
    assert "password_hash" not in r.json()

    r = login(client, email)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"

    me = client.get("/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_email_is_unique_and_case_insensitive(client):
    email = new_email()
    assert register(client, email).status_code == 201
    assert register(client, email.upper()).status_code == 409
    assert login(client, email.upper()).status_code == 200


def test_short_password_rejected(client):
    assert register(client, new_email(), "short").status_code == 422


def test_wrong_password_and_unknown_email_look_the_same(client):
    email = new_email()
    register(client, email)
    wrong = login(client, email, "wrong-password")
    unknown = login(client, new_email())
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # don't reveal which emails exist


@pytest.mark.parametrize(
    "method, path, kwargs",
    [
        ("get", "/auth/me", {}),
        ("get", "/documents", {}),
        ("post", "/chat", {"json": {"question": "hi"}}),
        ("post", "/documents", {"files": {"file": ("a.pdf", b"%PDF", "application/pdf")}}),
    ],
)
def test_protected_routes_require_token(client, method, path, kwargs):
    assert getattr(client, method)(path, **kwargs).status_code == 401


def _token(secret: str, **overrides) -> str:
    now = datetime.now(UTC)
    payload = {"sub": str(uuid.uuid4()), "iat": now, "exp": now + timedelta(minutes=5), **overrides}
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        _token("some-other-secret-that-is-also-long-enough-0123456789"),  # forged signature
        _token(settings.jwt_secret, exp=datetime.now(UTC) - timedelta(minutes=1)),  # expired
    ],
    ids=["garbage", "wrong-secret", "expired"],
)
def test_bad_tokens_rejected(client, token):
    assert client.get("/documents", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_users_cannot_access_each_others_documents(client):
    alice, bob = auth_headers(client), auth_headers(client)

    r = client.post("/documents", headers=alice, files={"file": ("a.pdf", blank_pdf(), "application/pdf")})
    assert r.status_code == 202
    doc_id = r.json()["id"]

    assert client.get("/documents", headers=bob).json() == []
    assert client.get(f"/documents/{doc_id}", headers=bob).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=bob).status_code == 404

    assert [d["id"] for d in client.get("/documents", headers=alice).json()] == [doc_id]
    assert client.delete(f"/documents/{doc_id}", headers=alice).status_code == 204
