# Phase 1 Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 백엔드 전체 구축 — JWT 인증, Google OAuth, KIS API 키 등록, 종목 시세 API, Redis 캐싱, Docker Compose 환경

**Architecture:** FastAPI + async SQLAlchemy (PostgreSQL 15) + Redis 7. 인증은 JWT Access Token (30분) + Refresh Token Rotation (7일, HttpOnly Cookie). KIS API 키는 AES-256-GCM으로 DB 암호화 저장. 시세는 pykrx 수집 후 Redis TTL 캐싱 (장중 30초, 장외 24시간).

**Tech Stack:** FastAPI 0.111, SQLAlchemy 2 + asyncpg, Alembic, python-jose, passlib[bcrypt], cryptography, slowapi, authlib, pykrx, sendgrid, redis[asyncio], pytest + pytest-asyncio

---

## File Map

```
FinalProject/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py          # 회원가입·로그인·OAuth·KIS 키 엔드포인트
│   │   │   └── stocks.py        # 종목 목록·검색·차트 엔드포인트
│   │   ├── middleware/
│   │   │   └── rate_limit.py    # slowapi Limiter 싱글턴
│   │   └── deps.py              # get_current_user 의존성
│   ├── services/
│   │   ├── market_service.py    # pykrx OHLCV 조회 + Redis 캐싱
│   │   ├── email_service.py     # SendGrid 이메일 발송
│   │   └── kis_service.py       # KIS 연결 테스트 (non-dev 전용)
│   ├── models/
│   │   └── user.py              # User, RefreshToken ORM 모델
│   ├── core/
│   │   ├── config.py            # pydantic-settings Settings
│   │   ├── security.py          # JWT·AES-256-GCM·bcrypt
│   │   ├── database.py          # async SQLAlchemy 엔진·세션
│   │   └── redis_client.py      # Redis 연결 풀
│   ├── main.py                  # FastAPI 앱, 라우터 등록, CORS, SessionMiddleware
│   ├── Dockerfile
│   └── requirements.txt
├── db/
│   └── migrations/              # Alembic (alembic.ini는 FinalProject 루트)
├── tests/
│   ├── conftest.py              # pytest fixtures (test DB, async client)
│   ├── test_security.py         # security.py 유닛 테스트
│   ├── test_auth.py             # /auth 통합 테스트
│   └── test_stocks.py          # /stocks 통합 테스트
├── pytest.ini
├── docker-compose.yml
└── .env.example
```

**의존성 순서:** Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8 → Task 9 → Task 10 → Task 11

---

## Task 1: Project Scaffold

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `backend/Dockerfile`
- Create: `backend/requirements.txt`
- Create: `pytest.ini`

- [ ] **Step 1: Create `docker-compose.yml`**

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-stocksense}
      POSTGRES_USER: ${POSTGRES_USER:-stocksense}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-stocksense}
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-stocksense}"]
      interval: 5s
      timeout: 5s
      retries: 5
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"

  backend:
    build: ./backend
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app

  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    depends_on:
      - redis
      - postgres
    env_file: .env

volumes:
  pgdata:
```

- [ ] **Step 2: Create `.env.example`**

```
APP_ENV=development
SECRET_KEY=change-this-in-production-min-32-chars
ENCRYPTION_KEY=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=

POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stocksense
POSTGRES_USER=stocksense
POSTGRES_PASSWORD=stocksense
DATABASE_URL=postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense

REDIS_URL=redis://localhost:6379/0

SENDGRID_API_KEY=
FROM_EMAIL=noreply@stocksense.ai

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

CORS_ORIGINS=http://localhost:5173
FRONTEND_URL=http://localhost:5173
```

- [ ] **Step 3: Create `backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

- [ ] **Step 4: Create `backend/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy[asyncio]==2.0.30
asyncpg==0.29.0
alembic==1.13.1
pydantic[email]==2.7.1
pydantic-settings==2.2.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
httpx==0.27.0
redis[asyncio]==5.0.4
celery==5.3.6
slowapi==0.1.9
pykrx==1.0.47
sendgrid==6.11.0
cryptography==42.0.7
authlib==1.3.0
python-multipart==0.0.9
pytest==8.2.0
pytest-asyncio==0.23.6
```

- [ ] **Step 5: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 6: Verify structure**

```bash
ls docker-compose.yml .env.example backend/Dockerfile backend/requirements.txt pytest.ini
```
Expected: 모두 존재, 오류 없음

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml .env.example backend/Dockerfile backend/requirements.txt pytest.ini
git commit -m "chore: project scaffold — Docker Compose, Dockerfile, requirements"
```

---

## Task 2: Core Layer (Config + Database + Redis)

**Files:**
- Create: `backend/core/__init__.py`
- Create: `backend/core/config.py`
- Create: `backend/core/database.py`
- Create: `backend/core/redis_client.py`

- [ ] **Step 1: Create `backend/core/__init__.py`** (빈 파일)

- [ ] **Step 2: Create `backend/core/config.py`**

```python
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_ENV: str = "development"
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ENCRYPTION_KEY: str = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "stocksense"
    POSTGRES_USER: str = "stocksense"
    POSTGRES_PASSWORD: str = "stocksense"
    DATABASE_URL: str = "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense"

    REDIS_URL: str = "redis://localhost:6379/0"

    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@stocksense.ai"

    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/auth/google/callback"

    CORS_ORIGINS: str = "http://localhost:5173"
    FRONTEND_URL: str = "http://localhost:5173"

    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    EMAIL_VERIFY_TOKEN_EXPIRE_MINUTES: int = 30

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 3: Create `backend/core/database.py`**

```python
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=(settings.APP_ENV == "development"))
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

- [ ] **Step 4: Create `backend/core/redis_client.py`**

```python
import redis.asyncio as aioredis

from .config import settings

_redis: aioredis.Redis = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None
```

- [ ] **Step 5: Commit**

```bash
git add backend/core/
git commit -m "feat: core layer — config, database, redis client"
```

---

## Task 3: Security Module

**Files:**
- Create: `backend/core/security.py`
- Create: `tests/__init__.py`
- Create: `tests/test_security.py`

- [ ] **Step 1: Create `tests/__init__.py`** (빈 파일)

- [ ] **Step 2: Create `tests/test_security.py`** (failing tests)**

```python
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
```

- [ ] **Step 3: Run to confirm FAIL**

```bash
cd backend && python -m pytest ../tests/test_security.py -v
```
Expected: `ImportError` — `core.security` 없음

- [ ] **Step 4: Create `backend/core/security.py`**

```python
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
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
    return base64.b64encode(nonce + ct).decode()


def decrypt_aes(encrypted: str) -> str:
    """AES-256-GCM 복호화. 입력: base64(12-byte nonce + ciphertext)"""
    key = base64.b64decode(settings.ENCRYPTION_KEY)
    aesgcm = AESGCM(key)
    data = base64.b64decode(encrypted)
    nonce, ct = data[:12], data[12:]
    return aesgcm.decrypt(nonce, ct, None).decode()
```

- [ ] **Step 5: Run to confirm PASS**

```bash
cd backend && python -m pytest ../tests/test_security.py -v
```
Expected: 8개 모두 PASS

- [ ] **Step 6: Commit**

```bash
git add backend/core/security.py tests/
git commit -m "feat: security module — JWT, bcrypt, AES-256-GCM (TDD)"
```

---

## Task 4: User Models + Alembic

**Files:**
- Create: `backend/models/__init__.py`
- Create: `backend/models/user.py`
- Create: `db/migrations/` (alembic init)

- [ ] **Step 1: Create `backend/models/__init__.py`**

```python
from .user import RefreshToken, User

__all__ = ["User", "RefreshToken"]
```

- [ ] **Step 2: Create `backend/models/user.py`**

```python
import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    is_verified = Column(Boolean, default=False)
    google_id = Column(String(255), unique=True, nullable=True)
    mode = Column(String(20), default="demo")

    kis_paper_key_enc = Column(Text, nullable=True)
    kis_paper_secret_enc = Column(Text, nullable=True)
    kis_paper_account_no = Column(String(20), nullable=True)
    kis_real_key_enc = Column(Text, nullable=True)
    kis_real_secret_enc = Column(Text, nullable=True)
    kis_real_account_no = Column(String(20), nullable=True)

    dark_mode = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token_hash = Column(String(255), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="refresh_tokens")


Index("idx_refresh_tokens_user", RefreshToken.user_id)
```

- [ ] **Step 3: Postgres 먼저 기동**

```bash
docker compose up -d postgres
```
Expected: postgres 컨테이너 healthy 상태

- [ ] **Step 4: Alembic 초기화**

FinalProject 루트에서:

```bash
pip install alembic asyncpg
alembic init db/migrations
```
Expected: `db/migrations/` 폴더 + `alembic.ini` 생성

- [ ] **Step 5: `alembic.ini` 수정**

`alembic.ini`의 `sqlalchemy.url` 줄을 주석 처리 (env.py에서 동적 로드):

```ini
# sqlalchemy.url = driver://user:pass@localhost/dbname
```

- [ ] **Step 6: `db/migrations/env.py` 전체 교체**

```python
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from core.database import Base
from models.user import RefreshToken, User  # noqa: F401 — registers models with Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense",
    )


def run_migrations_offline() -> None:
    context.configure(
        url=get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = get_url()
    connectable = async_engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
```

- [ ] **Step 7: 초기 마이그레이션 생성**

```bash
alembic revision --autogenerate -m "initial_schema"
```
Expected: `db/migrations/versions/xxxx_initial_schema.py` 생성

- [ ] **Step 8: 마이그레이션 적용**

```bash
alembic upgrade head
```
Expected: `Running upgrade -> xxxx, initial_schema`

- [ ] **Step 9: Commit**

```bash
git add backend/models/ db/ alembic.ini
git commit -m "feat: user/refresh_token models + alembic initial migration"
```

---

## Task 5: FastAPI App + Middleware + Test Scaffold

**Files:**
- Create: `backend/api/__init__.py`
- Create: `backend/api/routes/__init__.py`
- Create: `backend/api/middleware/__init__.py`
- Create: `backend/api/middleware/rate_limit.py`
- Create: `backend/api/deps.py`
- Create: `backend/main.py`
- Create: `tests/conftest.py`
- Create: `tests/test_auth.py` (stub)

- [ ] **Step 1: `__init__.py` 파일 3개 생성** (모두 빈 파일)

- `backend/api/__init__.py`
- `backend/api/routes/__init__.py`
- `backend/api/middleware/__init__.py`

- [ ] **Step 2: Create `backend/api/middleware/rate_limit.py`**

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
```

- [ ] **Step 3: Create `backend/api/deps.py`**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import decode_access_token
from models.user import User

bearer_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    user_id = decode_access_token(credentials.credentials)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user
```

- [ ] **Step 4: Create `backend/main.py`**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.middleware.rate_limit import limiter
from core.config import settings
from core.redis_client import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_redis()


app = FastAPI(title="StockSenseAI API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

(라우터는 Task 6, 11에서 추가)

- [ ] **Step 5: Test DB 생성**

```bash
docker compose up -d postgres
docker compose exec postgres psql -U stocksense -c "CREATE DATABASE stocksense_test;"
```
Expected: `CREATE DATABASE`

- [ ] **Step 6: Create `tests/conftest.py`**

```python
import os

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

os.environ["APP_ENV"] = "test"
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense_test",
)

from core.database import Base, get_db  # noqa: E402
from main import app  # noqa: E402

TEST_DB_URL = os.environ["DATABASE_URL"]


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_session):
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
```

- [ ] **Step 7: Create `tests/test_auth.py` (health check stub)**

```python
async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 8: 테스트 실행 확인**

```bash
cd backend && python -m pytest ../tests/test_auth.py::test_health -v
```
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add backend/api/ backend/main.py tests/conftest.py tests/test_auth.py
git commit -m "feat: fastapi app scaffold — middleware, deps, test fixtures"
```

---

## Task 6: Email Service + Register + Verify-Email

**Files:**
- Create: `backend/services/__init__.py`
- Create: `backend/services/email_service.py`
- Create: `backend/api/routes/auth.py` (register, verify-email)
- Modify: `backend/main.py` (auth 라우터 등록)
- Modify: `tests/test_auth.py`

- [ ] **Step 1: Failing tests 추가 (`tests/test_auth.py`)**

```python
from unittest.mock import patch


async def test_register_success(client):
    with patch("api.routes.auth.send_verification_email") as mock_email:
        response = await client.post(
            "/auth/register", json={"email": "test@example.com", "password": "password123"}
        )
    assert response.status_code == 201
    assert response.json()["message"] == "인증 이메일을 발송했습니다"
    mock_email.assert_called_once_with("test@example.com")


async def test_register_duplicate_email(client):
    with patch("api.routes.auth.send_verification_email"):
        await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass123"})
        response = await client.post(
            "/auth/register", json={"email": "dup@example.com", "password": "pass123"}
        )
    assert response.status_code == 409


async def test_verify_email_success(client):
    from core.security import create_email_token

    with patch("api.routes.auth.send_verification_email"):
        await client.post(
            "/auth/register", json={"email": "verify@example.com", "password": "pass123"}
        )
    token = create_email_token("verify@example.com")
    response = await client.post("/auth/verify-email", json={"token": token})
    assert response.status_code == 200


async def test_verify_email_invalid_token(client):
    response = await client.post("/auth/verify-email", json={"token": "totally-invalid"})
    assert response.status_code == 400
```

- [ ] **Step 2: Fail 확인**

```bash
cd backend && python -m pytest ../tests/test_auth.py -k "register or verify" -v
```
Expected: FAIL (라우터 없음)

- [ ] **Step 3: Create `backend/services/__init__.py`** (빈 파일)

- [ ] **Step 4: Create `backend/services/email_service.py`**

```python
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

from core.config import settings
from core.security import create_email_token


async def send_verification_email(email: str) -> None:
    token = create_email_token(email)
    verify_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

    if not settings.SENDGRID_API_KEY:
        return  # dev 환경 — 실제 발송 생략

    message = Mail(
        from_email=settings.FROM_EMAIL,
        to_emails=email,
        subject="StockSenseAI 이메일 인증",
        html_content=(
            f"<p>아래 링크를 클릭하여 이메일을 인증하세요 (30분 유효):</p>"
            f'<a href="{verify_url}">{verify_url}</a>'
        ),
    )
    SendGridAPIClient(settings.SENDGRID_API_KEY).send(message)
```

- [ ] **Step 5: Create `backend/api/routes/auth.py`**

```python
from datetime import datetime, timedelta, timezone

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from api.middleware.rate_limit import limiter
from core.config import settings
from core.database import get_db
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_email_token,
    encrypt_aes,
    hash_password,
    verify_password,
    verify_refresh_token,
)
from models.user import RefreshToken, User
from services.email_service import send_verification_email

router = APIRouter()

_REFRESH_COOKIE = "refresh_token"

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


# ── Pydantic 스키마 ───────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class VerifyEmailRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class KISKeyRequest(BaseModel):
    mode: str  # 'paper' | 'real'
    app_key: str
    app_secret: str
    account_no: str


class MeResponse(BaseModel):
    id: str
    email: str
    is_verified: bool
    mode: str
    dark_mode: bool


# ── 엔드포인트 ─────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 등록된 이메일입니다")

    user = User(email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    await db.commit()

    await send_verification_email(body.email)
    return {"message": "인증 이메일을 발송했습니다"}


@router.post("/verify-email")
async def verify_email(body: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    email = decode_email_token(body.token)
    if not email:
        raise HTTPException(status_code=400, detail="유효하지 않거나 만료된 토큰입니다")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    user.is_verified = True
    await db.commit()
    return {"message": "이메일 인증이 완료되었습니다"}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request, body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="이메일 인증이 필요합니다")

    raw_rt, hashed_rt = create_refresh_token()
    expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=user.id, token_hash=hashed_rt, expires_at=expires))
    await db.commit()

    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=raw_rt,
        httponly=True,
        secure=False,  # 프로덕션에서는 True
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/refresh")
async def refresh_token(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    raw_rt = request.cookies.get(_REFRESH_COOKIE)
    if not raw_rt:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.revoked == False,  # noqa: E712
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    tokens = result.scalars().all()
    matched = next((t for t in tokens if verify_refresh_token(raw_rt, t.token_hash)), None)
    if not matched:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    matched.revoked = True
    raw_new, hashed_new = create_refresh_token()
    expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=matched.user_id, token_hash=hashed_new, expires_at=expires))
    await db.commit()

    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=raw_new,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    return {"access_token": create_access_token(str(matched.user_id)), "token_type": "bearer"}


@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    raw_rt = request.cookies.get(_REFRESH_COOKIE)
    if raw_rt:
        result = await db.execute(
            select(RefreshToken).where(RefreshToken.revoked == False)  # noqa: E712
        )
        tokens = result.scalars().all()
        matched = next((t for t in tokens if verify_refresh_token(raw_rt, t.token_hash)), None)
        if matched:
            matched.revoked = True
            await db.commit()
    response.delete_cookie(_REFRESH_COOKIE)
    return {"message": "로그아웃되었습니다"}


@router.get("/google")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, settings.GOOGLE_REDIRECT_URI)


@router.get("/google/callback")
async def google_callback(
    request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=400, detail="Google OAuth 인증 실패")

    userinfo = token.get("userinfo", {})
    email = userinfo.get("email")
    google_id = userinfo.get("sub")
    if not email:
        raise HTTPException(status_code=400, detail="Google에서 이메일을 가져올 수 없습니다")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        user.google_id = google_id
        user.is_verified = True
    else:
        user = User(email=email, google_id=google_id, is_verified=True)
        db.add(user)
    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(str(user.id))
    return RedirectResponse(url=f"{settings.FRONTEND_URL}/oauth-callback#token={access_token}")


@router.get("/me", response_model=MeResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=str(current_user.id),
        email=current_user.email,
        is_verified=current_user.is_verified,
        mode=current_user.mode,
        dark_mode=current_user.dark_mode,
    )


@router.put("/api-key")
async def register_api_key(
    body: KISKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if body.mode not in ("paper", "real"):
        raise HTTPException(status_code=400, detail="mode는 'paper' 또는 'real'이어야 합니다")

    key_enc = encrypt_aes(body.app_key)
    secret_enc = encrypt_aes(body.app_secret)

    if body.mode == "paper":
        current_user.kis_paper_key_enc = key_enc
        current_user.kis_paper_secret_enc = secret_enc
        current_user.kis_paper_account_no = body.account_no
    else:
        current_user.kis_real_key_enc = key_enc
        current_user.kis_real_secret_enc = secret_enc
        current_user.kis_real_account_no = body.account_no

    if settings.APP_ENV != "development":
        from services.kis_service import test_kis_connection

        ok = await test_kis_connection(body.app_key, body.app_secret, body.mode)
        if not ok:
            raise HTTPException(status_code=400, detail="KIS API 키 검증에 실패했습니다")

    current_user.mode = body.mode
    await db.commit()
    return {"message": "KIS API 키가 등록되었습니다"}
```

- [ ] **Step 6: `main.py`에 auth 라우터 등록**

`backend/main.py` lifespan 아래에 추가:

```python
from api.routes import auth  # 파일 상단 import에 추가

app.include_router(auth.router, prefix="/auth", tags=["auth"])
```

- [ ] **Step 7: 테스트 실행**

```bash
cd backend && python -m pytest ../tests/test_auth.py -v
```
Expected: 5개 모두 PASS (health + 4 register/verify)

- [ ] **Step 8: Commit**

```bash
git add backend/services/ backend/api/routes/auth.py backend/main.py tests/test_auth.py
git commit -m "feat: email service + auth register/verify endpoints"
```

---

## Task 7: Auth Login + Refresh + Logout 테스트

**Files:**
- Modify: `tests/test_auth.py` (login, refresh, logout 테스트 추가)

(구현은 Task 6에서 `auth.py`에 모두 포함됨)

- [ ] **Step 1: Failing tests 추가**

```python
async def _register_and_verify(client, email: str, password: str = "pass1234"):
    with patch("api.routes.auth.send_verification_email"):
        await client.post("/auth/register", json={"email": email, "password": password})
    from core.security import create_email_token
    token = create_email_token(email)
    await client.post("/auth/verify-email", json={"token": token})


async def test_login_success(client):
    await _register_and_verify(client, "login@test.com")
    response = await client.post(
        "/auth/login", json={"email": "login@test.com", "password": "pass1234"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.cookies.get("refresh_token") is not None


async def test_login_wrong_password(client):
    await _register_and_verify(client, "wrongpw@test.com")
    response = await client.post(
        "/auth/login", json={"email": "wrongpw@test.com", "password": "badpass"}
    )
    assert response.status_code == 401


async def test_login_unverified_user(client):
    with patch("api.routes.auth.send_verification_email"):
        await client.post(
            "/auth/register", json={"email": "unverified@test.com", "password": "pass1234"}
        )
    response = await client.post(
        "/auth/login", json={"email": "unverified@test.com", "password": "pass1234"}
    )
    assert response.status_code == 403


async def test_refresh_token_rotation(client):
    await _register_and_verify(client, "refresh@test.com")
    await client.post("/auth/login", json={"email": "refresh@test.com", "password": "pass1234"})

    refresh_resp = await client.post("/auth/refresh")
    assert refresh_resp.status_code == 200
    assert "access_token" in refresh_resp.json()
    assert refresh_resp.cookies.get("refresh_token") is not None


async def test_logout_invalidates_refresh_token(client):
    await _register_and_verify(client, "logout@test.com")
    await client.post("/auth/login", json={"email": "logout@test.com", "password": "pass1234"})

    logout_resp = await client.post("/auth/logout")
    assert logout_resp.status_code == 200

    # 로그아웃 후 refresh는 401
    refresh_resp = await client.post("/auth/refresh")
    assert refresh_resp.status_code == 401
```

- [ ] **Step 2: 테스트 실행**

```bash
cd backend && python -m pytest ../tests/test_auth.py -k "login or refresh or logout" -v
```
Expected: 5개 모두 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_auth.py
git commit -m "test: auth login/refresh/logout integration tests"
```

---

## Task 8: Google OAuth + /me + KIS Key 테스트

**Files:**
- Modify: `tests/test_auth.py`

(구현은 Task 6의 `auth.py`에 모두 포함됨)

- [ ] **Step 1: Failing tests 추가**

```python
async def test_google_login_redirects(client):
    # GOOGLE_CLIENT_ID가 설정된 환경에서만 302 확인 가능
    # 미설정 시 authlib가 오류를 반환하므로 4xx도 허용
    response = await client.get("/auth/google", follow_redirects=False)
    assert response.status_code in (302, 307, 400, 422)
    if response.status_code in (302, 307):
        assert "accounts.google.com" in response.headers.get("location", "")


async def test_me_returns_user_info(client):
    await _register_and_verify(client, "me@test.com")
    login_resp = await client.post(
        "/auth/login", json={"email": "me@test.com", "password": "pass1234"}
    )
    token = login_resp.json()["access_token"]

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "me@test.com"
    assert data["mode"] == "demo"
    assert data["is_verified"] is True


async def test_me_unauthorized_without_token(client):
    response = await client.get("/auth/me")
    assert response.status_code == 403  # HTTPBearer returns 403 when header missing


async def test_register_kis_paper_key(client):
    await _register_and_verify(client, "kis@test.com")
    login_resp = await client.post(
        "/auth/login", json={"email": "kis@test.com", "password": "pass1234"}
    )
    token = login_resp.json()["access_token"]

    response = await client.put(
        "/auth/api-key",
        json={
            "mode": "paper",
            "app_key": "PXXXXXXXXXXXXXXXXXXX",
            "app_secret": "SXXXXXXXXXXXXXXXXXXX",
            "account_no": "12345678-01",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["message"] == "KIS API 키가 등록되었습니다"

    # mode가 paper로 업데이트됐는지 확인
    me_resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.json()["mode"] == "paper"
```

- [ ] **Step 2: 테스트 실행**

```bash
cd backend && python -m pytest ../tests/test_auth.py -k "google or me or kis" -v
```
Expected: 4개 모두 PASS

(`test_google_login_redirects`는 GOOGLE_CLIENT_ID가 빈 문자열이면 authlib가 오류를 낼 수 있음 — 이 경우 302/307 대신 400 응답 가능. 환경변수 없을 때 404 대신 적절한 오류 반환이면 PASS로 인정)

- [ ] **Step 3: KIS 서비스 stub 생성**

Create `backend/services/kis_service.py`:

```python
import httpx

_KIS_REAL_URL = "https://openapi.koreainvestment.com:9443"
_KIS_PAPER_URL = "https://openapivts.koreainvestment.com:29443"


async def test_kis_connection(app_key: str, app_secret: str, mode: str) -> bool:
    base_url = _KIS_PAPER_URL if mode == "paper" else _KIS_REAL_URL
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{base_url}/oauth2/tokenP",
                json={
                    "grant_type": "client_credentials",
                    "appkey": app_key,
                    "appsecret": app_secret,
                },
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code == 200 and bool(resp.json().get("access_token"))
        except Exception:
            return False
```

- [ ] **Step 4: 전체 auth 테스트 실행**

```bash
cd backend && python -m pytest ../tests/test_auth.py -v
```
Expected: 모든 auth 테스트 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/kis_service.py tests/test_auth.py
git commit -m "feat: KIS service stub + auth /me /api-key /google tests"
```

---

## Task 9: Market Service (pykrx + Redis Caching)

**Files:**
- Create: `backend/services/market_service.py`
- Create: `tests/test_stocks.py`

- [ ] **Step 1: Create `tests/test_stocks.py`** (failing)**

```python
import json
from unittest.mock import AsyncMock, patch


async def test_ohlcv_cache_miss_calls_pykrx(client):
    mock_data = [
        {"date": "20240101", "open": 70000, "high": 72000, "low": 69000, "close": 71000, "volume": 1000000}
    ]
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex = AsyncMock()

    with patch("services.market_service.get_ohlcv_from_pykrx", new_callable=AsyncMock, return_value=mock_data):
        with patch("services.market_service.get_redis", return_value=mock_redis):
            response = await client.get("/stocks/005930/chart?period=1m&interval=day")

    assert response.status_code == 200
    data = response.json()
    assert data["code"] == "005930"
    assert data["data"] == mock_data
    mock_redis.setex.assert_called_once()


async def test_ohlcv_cache_hit_skips_pykrx(client):
    cached = [
        {"date": "20240101", "open": 70000, "high": 72000, "low": 69000, "close": 71000, "volume": 1000000}
    ]
    mock_redis = AsyncMock()
    mock_redis.get.return_value = json.dumps(cached)

    with patch("services.market_service.get_redis", return_value=mock_redis):
        with patch("services.market_service.get_ohlcv_from_pykrx") as mock_pykrx:
            response = await client.get("/stocks/005930/chart?period=1m&interval=day")

    assert response.status_code == 200
    assert response.json()["data"] == cached
    mock_pykrx.assert_not_called()
```

- [ ] **Step 2: Fail 확인**

```bash
cd backend && python -m pytest ../tests/test_stocks.py -v
```
Expected: FAIL (stocks 라우터 없음)

- [ ] **Step 3: Create `backend/services/market_service.py`**

```python
import json
from datetime import datetime, timedelta

from pykrx import stock as pykrx_stock

from core.redis_client import get_redis


def _is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = now.hour * 100 + now.minute
    return 900 <= t <= 1530


async def get_ohlcv_from_pykrx(code: str, period: str, interval: str) -> list[dict]:
    period_days = {"1w": 7, "1m": 30, "3m": 90, "6m": 180, "1y": 365, "3y": 1095}
    freq_map = {"day": "d", "week": "w", "month": "m"}

    end = datetime.now()
    start = end - timedelta(days=period_days.get(period, 30))
    df = pykrx_stock.get_market_ohlcv_by_date(
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
        code,
        freq=freq_map.get(interval, "d"),
    )
    if df is None or df.empty:
        return []

    return [
        {
            "date": date.strftime("%Y%m%d"),
            "open": int(row.get("시가", 0)),
            "high": int(row.get("고가", 0)),
            "low": int(row.get("저가", 0)),
            "close": int(row.get("종가", 0)),
            "volume": int(row.get("거래량", 0)),
        }
        for date, row in df.iterrows()
    ]


async def get_ohlcv_cached(code: str, period: str, interval: str) -> list[dict]:
    redis = await get_redis()
    cache_key = f"ohlcv:{code}:{period}:{interval}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    data = await get_ohlcv_from_pykrx(code, period, interval)
    ttl = 30 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


async def get_stock_current_price(code: str) -> dict:
    redis = await get_redis()
    cache_key = f"price:{code}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    today = datetime.now().strftime("%Y%m%d")
    df = pykrx_stock.get_market_ohlcv_by_date(today, today, code)
    if df is None or df.empty:
        return {"code": code}

    row = df.iloc[-1]
    data = {
        "code": code,
        "close": int(row.get("종가", 0)),
        "open": int(row.get("시가", 0)),
        "high": int(row.get("고가", 0)),
        "low": int(row.get("저가", 0)),
        "volume": int(row.get("거래량", 0)),
    }
    ttl = 30 if _is_market_open() else 86400
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


async def get_stock_list(market: str, limit: int, page: int) -> list[dict]:
    redis = await get_redis()
    cache_key = f"stocklist:{market.lower()}"

    cached = await redis.get(cache_key)
    if cached:
        tickers = json.loads(cached)
    else:
        market_str = "KOSPI" if market.lower() == "kospi" else "KOSDAQ"
        tickers = list(pykrx_stock.get_market_ticker_list(market=market_str))
        await redis.setex(cache_key, 86400, json.dumps(tickers))

    start = (page - 1) * limit
    page_tickers = tickers[start : start + limit]
    return [{"code": t, "name": pykrx_stock.get_market_ticker_name(t)} for t in page_tickers]


async def search_stocks(query: str) -> list[dict]:
    redis = await get_redis()
    all_tickers: list[str] = []

    for market in ["kospi", "kosdaq"]:
        cache_key = f"stocklist:{market}"
        cached = await redis.get(cache_key)
        if cached:
            all_tickers.extend(json.loads(cached))
        else:
            tickers = list(pykrx_stock.get_market_ticker_list(market=market.upper()))
            await redis.setex(cache_key, 86400, json.dumps(tickers))
            all_tickers.extend(tickers)

    q = query.lower()
    results = []
    for code in all_tickers:
        name = pykrx_stock.get_market_ticker_name(code)
        if q in code.lower() or q in name.lower():
            results.append({"code": code, "name": name})
        if len(results) >= 20:
            break
    return results


async def get_indices() -> list[dict]:
    redis = await get_redis()
    cache_key = "indices"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    today = datetime.now().strftime("%Y%m%d")
    result = []
    for name, code in [("KOSPI", "1"), ("KOSDAQ", "2")]:
        try:
            df = pykrx_stock.get_index_ohlcv_by_date(today, today, code)
            if df is not None and not df.empty:
                row = df.iloc[-1]
                result.append({
                    "name": name,
                    "value": float(row.get("종가", 0)),
                    "change_rate": float(row.get("등락률", 0)),
                })
        except Exception:
            pass

    ttl = 30 if _is_market_open() else 3600
    await redis.setex(cache_key, ttl, json.dumps(result))
    return result
```

- [ ] **Step 4: 테스트 실행 (stocks 라우터 필요, Task 10에서 추가 후 재실행)**

일단 market_service만 import 오류 없는지 확인:

```bash
cd backend && python -c "from services.market_service import get_ohlcv_cached; print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add backend/services/market_service.py tests/test_stocks.py
git commit -m "feat: market service — pykrx OHLCV + Redis caching"
```

---

## Task 10: Stocks Routes + 전체 테스트

**Files:**
- Create: `backend/api/routes/stocks.py`
- Modify: `backend/main.py` (stocks 라우터 등록)
- Modify: `tests/test_stocks.py` (route 테스트 추가)

- [ ] **Step 1: Failing route tests 추가 (`tests/test_stocks.py`)**

```python
from unittest.mock import AsyncMock, patch


async def test_stock_list_endpoint(client):
    mock_list = [
        {"code": "005930", "name": "삼성전자"},
        {"code": "000660", "name": "SK하이닉스"},
    ]
    with patch("services.market_service.get_stock_list", new_callable=AsyncMock, return_value=mock_list):
        response = await client.get("/stocks?market=kospi&limit=2&page=1")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["code"] == "005930"


async def test_stock_search_endpoint(client):
    mock_results = [{"code": "005930", "name": "삼성전자"}]
    with patch("services.market_service.search_stocks", new_callable=AsyncMock, return_value=mock_results):
        response = await client.get("/stocks/search?q=삼성")
    assert response.status_code == 200
    assert response.json()[0]["code"] == "005930"


async def test_stock_detail_endpoint(client):
    mock_price = {
        "code": "005930",
        "close": 72000,
        "open": 71000,
        "high": 73000,
        "low": 70000,
        "volume": 5000000,
    }
    with patch(
        "services.market_service.get_stock_current_price",
        new_callable=AsyncMock,
        return_value=mock_price,
    ):
        response = await client.get("/stocks/005930")
    assert response.status_code == 200
    assert response.json()["code"] == "005930"
    assert response.json()["close"] == 72000


async def test_indices_endpoint(client):
    mock_indices = [{"name": "KOSPI", "value": 2700.5, "change_rate": 0.3}]
    with patch(
        "services.market_service.get_indices",
        new_callable=AsyncMock,
        return_value=mock_indices,
    ):
        response = await client.get("/stocks/indices")
    assert response.status_code == 200
    assert response.json()[0]["name"] == "KOSPI"
```

- [ ] **Step 2: Fail 확인**

```bash
cd backend && python -m pytest ../tests/test_stocks.py -k "endpoint" -v
```
Expected: FAIL (routes 없음)

- [ ] **Step 3: Create `backend/api/routes/stocks.py`**

```python
from fastapi import APIRouter, Request, Query

from api.middleware.rate_limit import limiter
from services import market_service

router = APIRouter()


@router.get("")
@limiter.limit("100/minute")
async def list_stocks(
    request: Request,
    market: str = Query("kospi", pattern="^(kospi|kosdaq)$"),
    limit: int = Query(50, ge=1, le=200),
    page: int = Query(1, ge=1),
):
    return await market_service.get_stock_list(market, limit, page)


@router.get("/search")
@limiter.limit("100/minute")
async def search_stocks(request: Request, q: str = Query(..., min_length=1)):
    return await market_service.search_stocks(q)


@router.get("/indices")
@limiter.limit("100/minute")
async def get_indices(request: Request):
    return await market_service.get_indices()


@router.get("/{code}/chart")
@limiter.limit("100/minute")
async def get_stock_chart(
    request: Request,
    code: str,
    period: str = Query("1m", pattern="^(1w|1m|3m|6m|1y|3y)$"),
    interval: str = Query("day", pattern="^(day|week|month)$"),
):
    data = await market_service.get_ohlcv_cached(code, period, interval)
    return {"code": code, "period": period, "interval": interval, "data": data}


@router.get("/{code}")
@limiter.limit("100/minute")
async def get_stock_detail(request: Request, code: str):
    return await market_service.get_stock_current_price(code)
```

**주의:** `/search`와 `/indices`는 `/{code}` 앞에 위치해야 함 (FastAPI 라우터 순서).

- [ ] **Step 4: stocks 라우터 `main.py`에 등록**

`backend/main.py` 수정:

```python
from api.routes import auth, stocks  # auth import에 stocks 추가

# 기존 auth 라우터 아래에 추가:
app.include_router(stocks.router, prefix="/stocks", tags=["stocks"])
```

- [ ] **Step 5: 전체 테스트 실행**

```bash
cd backend && python -m pytest ../tests/ -v
```
Expected: 모든 테스트 PASS

- [ ] **Step 6: 수동 스모크 테스트**

```bash
cd backend && uvicorn main:app --reload --port 8000
# 별도 터미널:
curl http://localhost:8000/health
curl "http://localhost:8000/stocks?market=kospi&limit=3"
curl http://localhost:8000/stocks/indices
```
Expected: 각각 JSON 응답 (pykrx가 데이터 가져오는 데 몇 초 소요)

- [ ] **Step 7: Commit**

```bash
git add backend/api/routes/stocks.py backend/main.py tests/test_stocks.py
git commit -m "feat: stocks routes — list, search, detail, chart, indices (TDD)"
```

---

## Task 11: 최종 통합 확인 + Docker 빌드

**Files:** 없음 (확인 작업)

- [ ] **Step 1: 전체 테스트 suite 실행**

```bash
cd backend && python -m pytest ../tests/ -v --tb=short
```
Expected: 모든 테스트 PASS, 0 failed

- [ ] **Step 2: Docker Compose 빌드 확인**

```bash
docker compose build backend
```
Expected: Successfully built

- [ ] **Step 3: Docker Compose 풀 스택 기동**

```bash
docker compose up -d postgres redis backend
docker compose logs backend --tail=20
```
Expected: `Application startup complete.`

- [ ] **Step 4: Swagger UI 확인**

브라우저에서 `http://localhost:8000/docs` 열기

확인 항목:
- `/auth/register` POST
- `/auth/login` POST
- `/auth/me` GET
- `/stocks` GET
- `/stocks/{code}/chart` GET

- [ ] **Step 5: 최종 commit + push**

```bash
git add -A
git status  # 커밋 안 된 파일 없는지 확인
git push origin hwang
```
Expected: hwang 브랜치에 모든 코드 push 완료
