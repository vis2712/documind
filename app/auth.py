"""Email + password auth issuing short-lived JWT access tokens (HS256)."""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from psycopg.errors import UniqueViolation
from pwdlib import PasswordHash
from pydantic import BaseModel, EmailStr, Field

from app.config import settings
from app.db import get_pool

ALGORITHM = "HS256"

router = APIRouter(prefix="/auth", tags=["auth"])
password_hasher = PasswordHash.recommended()  # Argon2id
# Used when the email doesn't exist, so a failed login takes the same time either way
# (otherwise response time reveals which emails are registered).
_DUMMY_HASH = password_hasher.hash("dummy-password-for-constant-time")

# tokenUrl powers the "Authorize" button in Swagger UI (/docs).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

INVALID_CREDENTIALS = HTTPException(
    status.HTTP_401_UNAUTHORIZED, "Incorrect email or password", headers={"WWW-Authenticate": "Bearer"}
)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds


def create_access_token(user_id: str) -> str:
    now = datetime.now(UTC)
    payload = {"sub": user_id, "iat": now, "exp": now + timedelta(minutes=settings.jwt_expire_minutes)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


async def current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """FastAPI dependency: returns the user id from a valid Bearer token, otherwise 401."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[ALGORITHM],  # pin the algorithm: never trust the token's own "alg" header
            options={"require": ["sub", "exp"]},
        )
    except jwt.InvalidTokenError:  # bad signature, expired, malformed, missing claims
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        ) from None
    return payload["sub"]


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    try:
        async with get_pool().connection() as conn:
            cur = await conn.execute(
                "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id, email, created_at",
                (body.email.lower(), password_hasher.hash(body.password)),
            )
            return await cur.fetchone()
    except UniqueViolation:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None


@router.post("/login", response_model=Token)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """OAuth2 password flow: form fields `username` (your email) and `password`."""
    async with get_pool().connection() as conn:
        cur = await conn.execute("SELECT id, password_hash FROM users WHERE email = %s", (form.username.lower(),))
        user = await cur.fetchone()

    if user is None:
        password_hasher.verify(form.password, _DUMMY_HASH)
        raise INVALID_CREDENTIALS
    if not password_hasher.verify(form.password, user["password_hash"]):
        raise INVALID_CREDENTIALS
    return Token(access_token=create_access_token(str(user["id"])), expires_in=settings.jwt_expire_minutes * 60)


@router.get("/me")
async def me(user_id: Annotated[str, Depends(current_user)]):
    async with get_pool().connection() as conn:
        cur = await conn.execute("SELECT id, email, created_at FROM users WHERE id = %s", (user_id,))
        user = await cur.fetchone()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User no longer exists")
    return user
