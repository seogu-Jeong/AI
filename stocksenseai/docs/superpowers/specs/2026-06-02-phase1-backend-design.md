# Phase 1 백엔드 설계 문서

**작성일:** 2026-06-02
**담당:** hygrenn (백엔드 + ML + 인프라)
**브랜치:** hwang

---

## 1. 범위

Phase 1 MVP 백엔드 전체 구조를 한 번에 구축한다.
- Docker Compose (PostgreSQL + Redis + Celery)
- FastAPI 앱 구조 + JWT 인증 (이메일 + Google OAuth)
- KIS API 키 등록 (AES-256-GCM 암호화, 개발환경에서는 연결 테스트 스킵)
- Alembic 마이그레이션
- Rate Limiting (slowapi)
- `/auth`, `/stocks` API

---

## 2. 디렉토리 구조

```
backend/
├── api/
│   ├── routes/
│   │   ├── auth.py             # 회원가입, 로그인, Google OAuth, KIS 키
│   │   └── stocks.py           # 종목 목록, 검색, 차트 데이터
│   ├── middleware/
│   │   ├── auth_middleware.py  # JWT 검증 의존성
│   │   └── rate_limit.py       # slowapi 설정
│   └── deps.py                 # 공통 의존성 (get_current_user 등)
├── services/
│   ├── market_service.py       # pykrx 시세 수집 + Redis 캐싱
│   └── email_service.py        # SendGrid 이메일 발송
├── models/
│   └── user.py                 # User, RefreshToken ORM 모델
├── core/
│   ├── config.py               # pydantic Settings (환경변수)
│   ├── security.py             # JWT 발급/검증, AES-256-GCM 암호화
│   ├── database.py             # async SQLAlchemy 세션
│   └── redis_client.py         # Redis 연결 풀
├── main.py                     # FastAPI 앱 + 라우터 등록
├── Dockerfile
└── requirements.txt

db/
└── migrations/                 # Alembic 마이그레이션 (루트 레벨)

docker-compose.yml
.env.example
```

---

## 3. DB 스키마

### users
```sql
CREATE TABLE users (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email            VARCHAR(255) UNIQUE NOT NULL,
    password_hash    VARCHAR(255),                    -- Google OAuth면 NULL 가능
    is_verified      BOOLEAN DEFAULT FALSE,
    google_id        VARCHAR(255) UNIQUE,             -- Google OAuth용
    mode             VARCHAR(20) DEFAULT 'demo',      -- 'demo' | 'paper' | 'real'
    kis_paper_key_enc    TEXT,                        -- AES-256-GCM 암호화
    kis_paper_secret_enc TEXT,
    kis_paper_account_no VARCHAR(20),
    kis_real_key_enc     TEXT,
    kis_real_secret_enc  TEXT,
    kis_real_account_no  VARCHAR(20),
    dark_mode        BOOLEAN DEFAULT TRUE,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
```

### refresh_tokens
```sql
CREATE TABLE refresh_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash  VARCHAR(255) NOT NULL,    -- bcrypt 해시 저장
    expires_at  TIMESTAMPTZ NOT NULL,
    revoked     BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_refresh_tokens_user ON refresh_tokens(user_id);
```

---

## 4. 인증 플로우

### 이메일 회원가입/로그인
```
POST /auth/register
  → 이메일 중복 확인
  → bcrypt 해싱
  → DB 저장 (is_verified=False)
  → SendGrid 인증 메일 발송 (30분 유효 토큰)
  → 201 반환

POST /auth/login
  → 이메일/비밀번호 검증
  → is_verified 확인
  → Access Token (JWT, 30분) 발급
  → Refresh Token (UUID, 7일) 생성 → bcrypt 해시 → DB 저장
  → Refresh Token: HttpOnly Secure Cookie로 전달
  → Access Token: JSON 응답 body

POST /auth/refresh
  → Cookie에서 Refresh Token 읽기
  → DB에서 hash 비교 + revoked 확인 + 만료 확인
  → 새 Access Token + 새 Refresh Token 발급 (Rotation)
  → 이전 Refresh Token revoked=True

POST /auth/logout
  → Cookie Refresh Token → DB revoked=True
  → Cookie 삭제
```

### Google OAuth
```
GET /auth/google
  → Google OAuth 동의 화면으로 리다이렉트

GET /auth/google/callback
  → Google에서 code 수신
  → Google API로 email, google_id, name 조회
  → users 테이블 upsert (email 기준)
  → JWT 발급 → 프론트 리다이렉트 (#token=xxx)
```

### KIS API 키 등록
```
PUT /auth/api-key
  body: { mode: 'paper'|'real', app_key, app_secret, account_no }
  → AES-256-GCM으로 app_key, app_secret 암호화
  → DB 저장
  → APP_ENV != 'development': KIS 토큰 발급 테스트 (잔고 조회 API)
  → 성공 시 user.mode 업데이트 (paper/real)
  → 200 { message, test_result }
```

---

## 5. API 엔드포인트 (Phase 1)

### /auth
| Method | Path | 설명 |
|---|---|---|
| POST | /auth/register | 회원가입 |
| POST | /auth/verify-email | 이메일 인증 |
| POST | /auth/login | 로그인 |
| POST | /auth/refresh | 토큰 갱신 |
| POST | /auth/logout | 로그아웃 |
| PUT | /auth/api-key | KIS 키 등록 |
| GET | /auth/me | 내 정보 |
| GET | /auth/google | Google OAuth 시작 |
| GET | /auth/google/callback | Google OAuth 콜백 |

### /stocks
| Method | Path | 설명 |
|---|---|---|
| GET | /stocks | 종목 목록 (?market=kospi\|kosdaq, limit, page) |
| GET | /stocks/search | 종목 검색 (?q) |
| GET | /stocks/{code} | 종목 상세 (현재가) |
| GET | /stocks/{code}/chart | OHLCV 차트 (?period, interval) |
| GET | /stocks/indices | 코스피/코스닥 지수 |

---

## 6. Docker Compose

```yaml
services:
  postgres:
    image: postgres:15-alpine
    environment: { POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD }
    volumes: pgdata:/var/lib/postgresql/data
    healthcheck: pg_isready

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru

  backend:
    build: ./backend
    depends_on: [postgres(healthy), redis]
    env_file: .env
    ports: 8000:8000
    volumes: ./backend:/app  # 개발용 hot reload

  celery:
    build: ./backend
    command: celery -A tasks worker --loglevel=info
    depends_on: [redis, postgres]
    env_file: .env
```

Nginx는 Phase 1 제외 (로컬 개발 단계).

---

## 7. 시세 캐싱 전략 (market_service.py)

```
GET /stocks/{code}/chart 요청
  → Redis 키: "ohlcv:{code}:{period}:{interval}"
  → 캐시 히트: 즉시 반환
  → 캐시 미스:
      장중 (09:00~15:30): pykrx 호출 → Redis TTL 30초
      장외: pykrx 호출 → Redis TTL 24시간
```

---

## 8. 보안

- JWT: `python-jose`, HS256, Access 30분
- Refresh Token: UUID → bcrypt 해시 저장, HttpOnly Secure Cookie
- KIS 키: AES-256-GCM, ENCRYPTION_KEY 환경변수
- Rate Limiting: 로그인 5회/분, 일반 API 100회/분 (slowapi)
- CORS: 개발 `http://localhost:5173`, 프로덕션 환경변수로 관리

---

## 9. KIS 레퍼런스 코드 활용 방침

`reference/` 폴더의 코드를 직접 복사하지 않는다.
활용하는 부분:
- TR ID 매핑 (매수: TTTC0802U, 매도: TTTC0801U 등)
- 실거래/모의 전환 로직 (TR ID 앞에 'V' 치환)
- KIS API URL 구조 및 응답 파싱 패턴

새로 짜는 부분:
- 동기 `requests` → 비동기 `httpx`
- 파일 기반 토큰 저장 → Redis 캐시
- 전역 변수 → 서비스 레이어 클래스

---

## 10. 개발/프로덕션 분기

```python
# core/config.py
APP_ENV: str = "development"  # development | production

# KIS 연결 테스트 스킵 (개발환경)
if settings.APP_ENV != "development":
    await kis_service.test_connection(app_key, app_secret, mode)
```
