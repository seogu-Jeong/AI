import base64
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_aes_key: bytes = base64.b64decode(settings.ENCRYPTION_KEY)
_aesgcm: AESGCM = AESGCM(_aes_key)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": user_id, "exp": expire}, settings.SECRET_KEY, algorithm="HS256")


def decode_access_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except JWTError:
        return None


def create_refresh_token() -> tuple[str, str]:
    """Returns (raw_token, bcrypt_hash). raw_token은 64자 hex."""
    raw = secrets.token_hex(32)
    hashed = pwd_context.hash(raw)
    return raw, hashed


def verify_refresh_token(raw: str, hashed: str) -> bool:
    return pwd_context.verify(raw, hashed)


def create_email_token(email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.EMAIL_VERIFY_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": email, "type": "email_verify", "exp": expire},
        settings.SECRET_KEY,
        algorithm="HS256",
    )


def decode_email_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        if payload.get("type") != "email_verify":
            return None
        return payload.get("sub")
    except JWTError:
        return None


def encrypt_aes(plaintext: str) -> str:
    """AES-256-GCM 암호화. 반환: base64(12-byte nonce + ciphertext)"""
    nonce = os.urandom(12)
    ct = _aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def decrypt_aes(encrypted: str) -> str:
    """AES-256-GCM 복호화. 입력: base64(12-byte nonce + ciphertext)"""
    try:
        data = base64.b64decode(encrypted)
        nonce, ct = data[:12], data[12:]
        return _aesgcm.decrypt(nonce, ct, None).decode()
    except Exception as exc:
        raise ValueError("AES-GCM decryption failed") from exc
