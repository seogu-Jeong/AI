import base64
import os

import pytest

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-unit-tests")
os.environ.setdefault("ENCRYPTION_KEY", base64.b64encode(os.urandom(32)).decode())

from core.security import (
    create_access_token,
    create_email_token,
    create_refresh_token,
    decrypt_aes,
    decode_access_token,
    decode_email_token,
    encrypt_aes,
    hash_password,
    settings,
    verify_password,
    verify_refresh_token,
)


def test_password_hash_and_verify():
    hashed = hash_password("mypassword")
    assert verify_password("mypassword", hashed)
    assert not verify_password("wrongpassword", hashed)


def test_access_token_roundtrip():
    token = create_access_token("user-123")
    assert decode_access_token(token) == "user-123"


def test_access_token_invalid_returns_none():
    assert decode_access_token("invalid.token.value") is None


def test_refresh_token_roundtrip():
    raw, hashed = create_refresh_token()
    assert len(raw) == 64  # token_hex(32) = 64 hex chars
    assert verify_refresh_token(raw, hashed)
    assert not verify_refresh_token("wrong-token", hashed)


def test_email_token_roundtrip():
    token = create_email_token("user@example.com")
    assert decode_email_token(token) == "user@example.com"


def test_email_token_wrong_type_returns_none():
    from datetime import datetime, timedelta, timezone

    from jose import jwt

    from core.security import settings

    payload = {
        "sub": "user@example.com",
        "type": "password_reset",
        "exp": datetime.now(timezone.utc) + timedelta(minutes=30),
    }
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    assert decode_email_token(token) is None


def test_aes_encrypt_decrypt_roundtrip():
    original = "sensitive-api-key-value"
    encrypted = encrypt_aes(original)
    assert encrypted != original
    assert decrypt_aes(encrypted) == original


def test_aes_encrypt_produces_different_ciphertexts():
    # 랜덤 nonce로 동일 평문이어도 매번 다른 암호문
    ct1 = encrypt_aes("same-value")
    ct2 = encrypt_aes("same-value")
    assert ct1 != ct2
    assert decrypt_aes(ct1) == decrypt_aes(ct2) == "same-value"


def test_aes_decrypt_rejects_tampered_ciphertext():
    import base64 as b64
    encrypted = encrypt_aes("original-value")
    raw = b64.b64decode(encrypted)
    # Flip a byte in the ciphertext body (after the 12-byte nonce)
    tampered = raw[:20] + bytes([raw[20] ^ 0xFF]) + raw[21:]
    with pytest.raises(ValueError):
        decrypt_aes(b64.b64encode(tampered).decode())


def test_access_token_expired_returns_none():
    from datetime import datetime, timedelta, timezone
    from jose import jwt
    payload = {"sub": "user-expired", "exp": datetime.now(timezone.utc) - timedelta(seconds=1)}
    token = jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")
    assert decode_access_token(token) is None
