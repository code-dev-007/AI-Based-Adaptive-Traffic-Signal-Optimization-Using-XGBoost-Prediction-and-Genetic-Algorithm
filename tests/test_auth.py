import pytest
from fastapi import HTTPException

from backend import auth


def test_password_is_hashed_and_verified():
    stored = auth.hash_password("secret123")
    assert "secret123" not in stored and stored.startswith("pbkdf2_sha256$")
    assert auth.verify_password("secret123", stored)
    assert not auth.verify_password("wrong-password", stored)
    assert auth.hash_password("secret123") != stored            # fresh random salt every time


def test_token_round_trip():
    assert auth.verify_token(auth.create_token("sanat")) == "sanat"


def test_tampered_or_expired_tokens_are_rejected():
    token = auth.create_token("sanat")
    payload, signature = token.split(".")
    assert auth.verify_token(f"{payload}.{signature[:-2]}xx") is None
    forged_payload = auth._b64(b'{"sub": "admin", "exp": 9999999999}')       # claims to be admin, reuses a real signature
    assert auth.verify_token(f"{forged_payload}.{signature}") is None
    assert auth.verify_token(auth.create_token("sanat", hours=-1)) is None   # already expired
    assert auth.verify_token("garbage") is None and auth.verify_token("") is None


def test_token_signed_with_another_key_is_rejected(monkeypatch):
    token = auth.create_token("sanat")
    monkeypatch.setenv("TRAFFIC_AI_SECRET", "a-different-key")
    assert auth.verify_token(token) is None


def test_require_user_needs_a_bearer_token():
    assert auth.require_user(f"Bearer {auth.create_token('sanat')}") == "sanat"
    for bad in (None, "", "Bearer nonsense", auth.create_token("sanat")):   # last one has no 'Bearer ' prefix
        with pytest.raises(HTTPException) as error:
            auth.require_user(bad)
        assert error.value.status_code == 401
