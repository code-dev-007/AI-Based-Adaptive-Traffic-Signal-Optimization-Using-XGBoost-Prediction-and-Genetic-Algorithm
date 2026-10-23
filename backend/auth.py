"""Password hashing and signed login tokens, using only the Python standard library.

Passwords are stored as PBKDF2-SHA256 hashes (never as plain text). A login token is
"<payload>.<signature>": the payload holds the username and an expiry time, and the
signature is an HMAC with a secret key, so a token cannot be edited or forged.
"""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

from fastapi import Header, HTTPException

ITERATIONS = 200_000
TOKEN_HOURS = 8
KEY_FILE = Path(__file__).resolve().parents[1] / "data" / ".secret_key"


def _secret() -> bytes:
    """Signing key: TRAFFIC_AI_SECRET if set, otherwise a random key saved once in data/.secret_key."""
    from_env = os.getenv("TRAFFIC_AI_SECRET")
    if from_env:
        return from_env.encode()
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(secrets.token_hex(32))
    return KEY_FILE.read_text().strip().encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), ITERATIONS).hex()
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, iterations, salt, digest = stored.split("$")
        check = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), int(iterations)).hex()
    except ValueError:
        return False
    return hmac.compare_digest(check, digest)


def create_token(username: str, hours: float = TOKEN_HOURS) -> str:
    payload = _b64(json.dumps({"sub": username, "exp": int(time.time() + hours * 3600)}).encode())
    signature = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}"


def verify_token(token: str) -> str | None:
    """Return the username if the token is genuine and not expired, otherwise None."""
    try:
        payload, signature = token.split(".")
        expected = _b64(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        data = json.loads(_unb64(payload))
    except (ValueError, KeyError):
        return None
    return data["sub"] if data.get("exp", 0) > time.time() else None


def require_user(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency: the request must carry 'Authorization: Bearer <token>'."""
    username = verify_token(authorization[7:]) if authorization and authorization.startswith("Bearer ") else None
    if username is None:
        raise HTTPException(status_code=401, detail="Not signed in, or the session has expired. Please sign in again.")
    return username
