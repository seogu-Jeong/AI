# StockSenseAI 백엔드 코드 검수 결과

> 작성일: 2026-06-03
> 검수 대상 브랜치: `hwang`
> 검수자: seogu-Jeong (프론트엔드)

---

## 🔴 심각 — 즉시 수정 필요

### 1. API 명세 미구현 엔드포인트

CLAUDE.md에 정의된 엔드포인트인데 `backend/api/routes/` 에 구현이 없습니다.
프론트엔드 연동 시 바로 막히는 항목들입니다.

| 엔드포인트 | 파일 | 상태 |
|---|---|---|
| `GET /stocks/{code}/orderbook` | `routes/stocks.py` | ❌ 미구현 |
| `GET /stocks/{code}/trades` | `routes/stocks.py` | ❌ 미구현 |
| `DELETE /auth/api-key` | `routes/auth.py` | ❌ 미구현 |

**프론트엔드 영향:**
- 우측 호가창(`OrderBook` 컴포넌트)이 실제 API 연동 시 404 오류 발생
- 실시간 체결 데이터 수신 불가
- KIS API 키 삭제 기능 동작 불가

**요청:** 위 3개 엔드포인트를 우선 구현해 주세요.

---

## 🟡 주의 — 수정 권장

### 2. config.py — 하드코딩된 기본값

```python
# 현재 코드 (위험)
SECRET_KEY = "dev-secret-key-change-in-production"
ENCRYPTION_KEY = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
```

프로덕션 배포 시 `.env` 미설정이면 그대로 사용될 수 있습니다.
아래 검증 코드 추가를 권장합니다:

```python
if APP_ENV == "production":
    assert SECRET_KEY != "dev-secret-key-change-in-production", "SECRET_KEY must be set"
    assert ENCRYPTION_KEY != "AAAAAAAAAAAAAAAAAAA...=", "ENCRYPTION_KEY must be set"
```

---

### 3. config.py — DATABASE_URL 재생성 로직

`assemble_db_url` 함수에서 `DATABASE_URL` 환경변수가 있을 때도 재생성될 수 있습니다.
로직을 점검해 주세요.

---

### 4. tasks.py — Celery 태스크 함수 미구현

```python
# tasks.py 에 실제 태스크 함수가 없음
```

프론트엔드 `BacktestTab`에서 `/backtest/run` API를 호출하면 Celery 태스크가 실행되어야 하는데,
태스크 함수 자체가 없어서 동작하지 않습니다.

백테스팅 Celery 태스크 구현이 필요합니다.

---

### 5. email_service.py — 실패 시 로깅 없음

```python
# 현재 코드
except Exception:
    pass  # 이메일 발송 실패해도 아무것도 기록 안 됨
```

발송 실패 여부를 알 수 없습니다. 최소한 로깅 추가를 권장합니다:

```python
except Exception as e:
    logger.error(f"Email send failed: {e}")
```

---

### 6. market_service.py — pykrx 예외 처리 로깅 없음

`get_stock_list()`, `search_stocks()`, `get_indices()` 등에서 pykrx 호출 실패 시
`pass`로 무시하고 있어 운영 중 디버깅이 어렵습니다. 로깅 추가를 권장합니다.

---

### 7. stocks.py — 에러 응답 형식 불일치

종목 상세 조회 시 데이터가 없으면 `{"code": code}` 형태로 반환합니다.
다른 엔드포인트는 `{"detail": "..."}` 형식이라 프론트엔드에서 일관성 있게 처리하기 어렵습니다.
404 HTTPException으로 통일해 주세요.

---

### 8. auth.py — 개발 환경에서 KIS API 키 검증 스킵

```python
if APP_ENV != "development":
    test_kis_connection()  # 개발 환경에서는 검증 안 함
```

개발 중 오타 입력된 키가 저장될 수 있습니다. 개발 환경에서도 포맷 검증 정도는 추가를 권장합니다.

---

### 9. requirements.txt — 상한 버전 미지정

```
pykrx>=1.2.8          # 상한 없음 → 메이저 업데이트 시 호환 깨질 수 있음
pytest-asyncio>=1.4.0  # 상한 없음
```

아래처럼 상한을 지정하는 것을 권장합니다:

```
pykrx>=1.2.8,<2.0.0
pytest-asyncio>=1.4.0,<2.0.0
```

---

## ✅ 잘 구현된 부분

- **JWT + Refresh Token 보안** — selector 기반 O(1) 조회, secure cookie 처리 우수
- **AES-256-GCM 암호화** — nonce 12byte, KIS API 키 암호화 올바르게 구현
- **Redis 캐싱 전략** — 장중 30초 / 장후 86400초 TTL 적절
- **DB 마이그레이션** — `users`, `refresh_tokens` 스키마 정상, 인덱스 설정 완료
- **테스트 커버리지** — auth, stocks 통합 테스트 포괄적으로 작성됨
- **Rate Limiting** — SlowAPI 기반 미들웨어 정상 설정
- **비동기 구조** — AsyncSession, asyncpg, asyncio 패턴 전반적으로 올바름

---

## 우선순위 요약

| 우선순위 | 항목 |
|---|---|
| 🔴 즉시 | `/stocks/{code}/orderbook`, `/stocks/{code}/trades`, `DELETE /auth/api-key` 구현 |
| 🔴 즉시 | Celery 백테스트 태스크 함수 구현 |
| 🟡 권장 | 프로덕션 환경 SECRET_KEY/ENCRYPTION_KEY 검증 |
| 🟡 권장 | email_service, market_service 로깅 추가 |
| 🟡 권장 | stocks.py 에러 응답 형식 통일 |
| 🟢 나중 | requirements.txt 상한 버전 지정 |

---

> 궁금한 점 있으면 카톡 주세요!
