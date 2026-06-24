# StockSenseAI — Phase별 진행 현황

**최종 업데이트:** 2026-06-04
**레포:** https://github.com/hygrenn/FinalProject | **브랜치:** `hwang` (백엔드), `seogu-Jeong` (프론트)

---

## 전체 Phase 구성

| Phase | 주제 | 담당 | 상태 |
|---|---|---|---|
| Phase 1 | 인증 + 기본 시세 API + 인프라 | hygrenn | ✅ 완료 |
| Phase 2 | 실시간 시세 + WebSocket + 차트 고도화 | hygrenn | ✅ 완료 |
| Phase 3 | AI 예측 + 시그널 + 패턴 인식 | hygrenn | ✅ 완료 |
| Phase 4 | 거래 + 포트폴리오 + 시뮬레이터 + 리스크 | hygrenn | 🔄 진행 중 (4-A 완료) |
| 프론트 전체 | 레이아웃 → 차트 → AI UI → 거래 UI | seogu-Jeong | 🔲 진행 중 |

---

## Phase 1 — 인증 + 기본 시세 API + 인프라 ✅

**완료일:** 2026-06-02 | **테스트:** 30/30 passing | **커밋:** `ec11bf5`

### 인프라

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| Docker Compose (postgres 15, redis 7, backend, celery) | `docker-compose.yml` | ✅ |
| PostgreSQL healthcheck + celery service_healthy 의존성 | `docker-compose.yml` | ✅ |
| Alembic 비동기 마이그레이션 (`async_engine_from_config`) | `db/migrations/` | ✅ |
| Celery stub (`tasks.py`) | `backend/tasks.py` | ✅ |
| `.env.example` (전체 환경변수 문서화) | `.env.example` | ✅ |

### 코어 레이어

| 컴포넌트 | 파일 | 비고 |
|---|---|---|
| pydantic-settings 설정 (`DATABASE_URL` 자동 조합 validator 포함) | `backend/core/config.py` | ✅ |
| async SQLAlchemy 엔진 + `get_db` | `backend/core/database.py` | ✅ |
| Redis 연결 풀 (lazy singleton) | `backend/core/redis_client.py` | ✅ |
| JWT (HS256, 30분) + Refresh Token (bcrypt hash, 7일) | `backend/core/security.py` | ✅ |
| AES-256-GCM 암호화/복호화 (모듈 로드 시 키 사전계산) | `backend/core/security.py` | ✅ |
| 이메일 인증 토큰 (type claim 포함, 30분) | `backend/core/security.py` | ✅ |

### DB 모델 & 마이그레이션

| 컴포넌트 | 파일 | 비고 |
|---|---|---|
| `users` 테이블 (UUID PK, CheckConstraint on mode, nullable=False 명시) | `backend/models/user.py` | ✅ |
| `refresh_tokens` 테이블 (selector 컬럼 + 복합 인덱스) | `backend/models/user.py` | ✅ |
| Initial schema 마이그레이션 | `db/migrations/versions/*_initial_schema.py` | ✅ |
| Selector 컬럼 추가 마이그레이션 | `db/migrations/versions/*_add_refresh_token_selector.py` | ✅ |

### 인증 API (`/auth`)

| 엔드포인트 | 설명 | 상태 |
|---|---|---|
| `POST /auth/register` | 이메일 가입, 중복 체크(409), 인증 메일 발송 | ✅ |
| `POST /auth/verify-email` | JWT 토큰으로 `is_verified=True` | ✅ |
| `POST /auth/login` | bcrypt 검증, Refresh Token Rotation, HttpOnly Cookie | ✅ |
| `POST /auth/refresh` | selector 인덱스 조회 → bcrypt 검증 → 토큰 교체 | ✅ |
| `POST /auth/logout` | Refresh Token revoked=True, 쿠키 삭제 | ✅ |
| `GET /auth/me` | JWT 인증 후 내 정보 반환 | ✅ |
| `PUT /auth/api-key` | KIS 키 AES-256-GCM 암호화 저장, 개발환경 연결 테스트 스킵 | ✅ |
| `GET /auth/google` | Google OAuth 동의 화면 리다이렉트 | ✅ |
| `GET /auth/google/callback` | upsert by email, JWT → 프론트 리다이렉트 | ✅ |

**보안 처리 포인트:**
- Refresh Token: `selector(16자) + bcrypt(verifier)` 구조 → O(1) DB 조회
- Cookie: `secure=True` (APP_ENV != development), `httponly=True`, `samesite=lax`
- Rate Limit: 로그인 5회/분, 일반 API 100회/분 (slowapi default_limits)
- KIS 키: `Literal["paper", "real"]` Pydantic 검증 + AES-256-GCM 암호화

### 시세 API (`/stocks`)

| 엔드포인트 | 설명 | 상태 |
|---|---|---|
| `GET /stocks` | 코스피/코스닥 종목 목록 (페이지네이션) | ✅ |
| `GET /stocks/search` | 종목명/코드 검색 (최대 20건) | ✅ |
| `GET /stocks/indices` | 코스피/코스닥 지수 | ✅ |
| `GET /stocks/{code}/chart` | OHLCV 차트 (period: 1w~3y, interval: day/week/month) | ✅ |
| `GET /stocks/{code}` | 종목 현재가 | ✅ |

**캐싱 전략:**
- 장중 (평일 09:00~15:30): Redis TTL 30초
- 장외: Redis TTL 24시간 (차트), 3600초 (지수)
- 종목 목록: Redis TTL 24시간

### 미들웨어 & 공통

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| JWT `get_current_user` (명시적 UUID 파싱 포함) | `backend/api/deps.py` | ✅ |
| slowapi Limiter (default_limits 100/min) | `backend/api/middleware/rate_limit.py` | ✅ |
| CORS + SessionMiddleware + SlowAPIMiddleware | `backend/main.py` | ✅ |
| SendGrid 이메일 서비스 (`asyncio.to_thread` + 오류 무시) | `backend/services/email_service.py` | ✅ |
| KIS 연결 테스트 stub (httpx async) | `backend/services/kis_service.py` | ✅ |
| pykrx OHLCV 수집 + Redis 캐싱 | `backend/services/market_service.py` | ✅ |

### 테스트

| 파일 | 테스트 수 | 커버리지 |
|---|---|---|
| `tests/test_security.py` | 10 | JWT, bcrypt, AES-256-GCM 유닛 |
| `tests/test_auth.py` | 14 | 전체 `/auth` 통합 (register→verify→login→refresh→logout→me→kis) |
| `tests/test_stocks.py` | 6 | 라우터/서비스 mock 기반 통합 (Redis 캐시 hit/miss, 각 stocks 엔드포인트) |
| **합계** | **30** | **30/30 passing** |

> 참고: stocks 테스트는 pykrx/Redis 실제 연동이 아닌 mock 기반 통합 테스트입니다. 별도 Redis/pykrx smoke test는 추후 추가 예정.

---

## Phase 2 — 실시간 시세 + WebSocket ✅

**목표:** KIS WebSocket 연동으로 실시간 체결/호가 데이터 스트리밍

**완료일:** 2026-06-03 | **테스트:** 52 passed

### 구현 완료 항목

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| KIS OAuth 토큰 캐시 (access_token + approval_key) | `backend/services/kis_token_service.py` | ✅ |
| KIS REST 호가·체결·분봉 | `backend/services/kis_market_service.py` | ✅ |
| KIS WebSocket Pool (41종목 배치, Redis Pub/Sub) | `backend/services/websocket_service.py` | ✅ |
| SSE 스트리밍 `/ws/stocks/{code}` | `backend/api/routes/realtime.py` | ✅ |
| `GET /stocks/{code}/orderbook` | `backend/api/routes/stocks.py` | ✅ |
| `GET /stocks/{code}/trades` | `backend/api/routes/stocks.py` | ✅ |
| 분봉 차트 (1min/5min/15min/1h) + period 1d | `backend/api/routes/stocks.py` | ✅ |

---

## Phase 3 — AI 예측 + 시그널 + 패턴 인식 ✅

**완료일:** 2026-06-04 | **테스트:** 65 passed (통합) + 23 passed (유닛)

**목표:** LSTM 모델로 5일 예측, 기술적 지표 기반 종합 시그널, 캔들 패턴 감지

> **주의:** 모델 구조·추론 코드는 구현 완료. 실제 학습 가중치(`backend/ml/weights/`)는 미포함 — 별도 학습 필요.
> `/ai/{code}/predict`는 가중치 없이 빈 예측값을 반환합니다.

### 구현 완료 항목

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| LSTM 모델 | `backend/ml/model.py` | ✅ |
| 피처 엔지니어링 (13개) | `backend/ml/features.py` | ✅ |
| 학습 스크립트 (코스피 상위 100) | `backend/ml/train.py` | ✅ |
| 추론 (MC Dropout 50회) | `backend/ml/predict.py` | ✅ |
| 유사 패턴 매칭 Top 5 | `backend/ml/pattern_matcher.py` | ✅ |
| AI 서비스 (지표 40% + LSTM 60%) | `backend/services/ai_service.py` | ✅ |
| 패턴 서비스 (14종) | `backend/services/pattern_service.py` | ✅ |
| Celery AI 태스크 | `backend/tasks/ai_tasks.py` | ✅ |
| AI 라우터 8개 엔드포인트 | `backend/api/routes/ai.py` | ✅ |
| APScheduler (15:35 KST) | `backend/main.py` | ✅ |
| Phase 3+4 DB 마이그레이션 | `db/migrations/versions/` | ✅ |

### AI API 엔드포인트 (`/ai`)

| 엔드포인트 | 설명 |
|---|---|
| `GET /ai/{code}/signal` | 종합 AI 시그널 (BUY/HOLD/SELL + score) |
| `GET /ai/{code}/predict` | LSTM 5일 예측 (bullish/base/bearish) |
| `GET /ai/{code}/indicators` | RSI, MACD, BB, MA 원시값 |
| `GET /ai/{code}/patterns` | 감지된 캔들 패턴 목록 |
| `GET /ai/{code}/similar` | 유사 패턴 히스토리 Top 5 |
| `GET /ai/{code}/multiframe` | 일봉/주봉/월봉 멀티타임프레임 시그널 |
| `GET /ai/top-picks` | AI BUY 시그널 상위 종목 (스크리너) |
| `GET /ai/signals/history/{code}` | 시그널 변경 이력 최근 30일 |

**Rate limit:** AI 엔드포인트 20회/분 (TRD 8.3)

---

## Phase 4 — 거래 + 포트폴리오 + 시뮬레이터 + 리스크

### Phase 4-A — 거래 + 포트폴리오 + 리스크 ✅

**완료일:** 2026-06-04 | **테스트:** 81 passed

### 구현 완료 항목

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| KIS 서비스 (완전판) | `backend/services/kis_service.py` | ✅ |
| 리스크 서비스 | `backend/services/risk_service.py` | ✅ |
| 체결 폴링 태스크 | `backend/tasks/order_tasks.py` | ✅ |
| 이메일 태스크 | `backend/tasks/email_tasks.py` | ✅ |
| 거래 API | `backend/api/routes/trades.py` | ✅ |
| 포트폴리오 API | `backend/api/routes/portfolio.py` | ✅ |
| 리스크 설정 API | `backend/api/routes/risk.py` | ✅ |
| 알림 설정 API | `backend/api/routes/alerts.py` | ✅ |
| 모드 전환 | `PUT /auth/mode` | ✅ |
| APScheduler 추가 스케줄 | `backend/main.py` | ✅ |
| DB 마이그레이션 v8 | enforce_hard_stop, notification_email | ✅ |

### Phase 4-B — 백테스팅 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 백테스팅 엔진 | `backend/services/backtest_service.py` | ✅ |
| 백테스팅 API | `backend/api/routes/backtest.py` | ✅ |
| 테스트 | `tests/test_backtest.py` (3 passed) | ✅ |

### Phase 4-C — 투자 시뮬레이터 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| price_cache 마이그레이션 | `db/migrations/versions/a7b8c9d0e1f2_add_price_cache.py` | ✅ |
| PriceCache 모델 | `backend/models/price_cache.py` | ✅ |
| 시뮬레이터 엔진 | `backend/services/simulator_service.py` | ✅ |
| 시뮬레이터 API | `backend/api/routes/simulate.py` | ✅ |
| 테스트 | `tests/test_simulate.py` (8 passed) | ✅ |

### Phase 4-D — 관심종목 + 알림 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 관심종목 CRUD API | `backend/api/routes/watchlist.py` | ✅ |
| 목표가 알림 | `backend/tasks/email_tasks.py` (check_price_alerts) | ✅ |
| 일일 손실 자동 차단 | `backend/tasks/email_tasks.py` (check_daily_loss) | ✅ |
| 테스트 | `tests/test_watchlist.py` (6) + `tests/test_alert_tasks.py` (5) | ✅ |

---

## 프론트엔드 (seogu-Jeong 담당) 🔲

**브랜치:** `seogu-Jeong` | **스택:** React 18 + TypeScript, Vite, shadcn/ui, Tailwind v4, Zustand, Axios

| 화면 | 설명 | 상태 |
|---|---|---|
| 랜딩 + 로그인/회원가입 | Google OAuth + 이메일 인증 UI | 🔲 |
| 메인 차트 화면 | Lightweight Charts 캔들스틱, AI 예측 오버레이 | 🔲 |
| 기술적 지표 패널 | RSI, MACD, 볼린저밴드 | 🔲 |
| AI 시그널 패널 | BUY/HOLD/SELL 스코어, 유사 패턴 | 🔲 |
| 종목 검색/관심종목 | 검색바, 관심종목 그룹 | 🔲 |
| 거래 패널 | 매수/매도 폼, 주문 현황 | 🔲 |
| 포트폴리오 화면 | 보유 종목, 수익률 차트 (Recharts) | 🔲 |
| 투자 시뮬레이터 | ifibought UX 기반 | 🔲 |
| 설정 화면 | KIS 키 등록, 리스크 설정, 알림 | 🔲 |

**백엔드 연동 전까지 mock 데이터로 개발.**
API Base: `http://localhost:8000` (개발), 환경변수: `VITE_API_BASE`

---

## 알려진 제약 / TODO

| 항목 | 내용 |
|---|---|
| Google OAuth refresh token | OAuth 콜백은 spec에 따라 `#token=xxx` redirect만 발급 (refresh token 미발급) |
| pykrx 1.2.8 KRX 로그인 | 환경변수 `KRX_ID`, `KRX_PW` 미설정 시 경고 출력 (데이터 조회는 정상 동작) |
| Docker 빌드 검증 | 로컬 Docker 미설치로 이미지 빌드 미검증 (요구사항 파일은 완비) |
| Nginx | Phase 1 제외, Phase 4 이후 프로덕션 배포 시 추가 |
