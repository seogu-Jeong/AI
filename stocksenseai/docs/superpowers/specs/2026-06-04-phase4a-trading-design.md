# Phase 4-A — 거래 + 포트폴리오 + 리스크 설계

**작성일:** 2026-06-04
**담당:** hygrenn (백엔드)
**레퍼런스:** TRD 섹션 5.4~5.6, 7, 9

---

## 1. 결정 사항 요약

| 항목 | 결정 |
|---|---|
| KIS 서비스 구조 | 함수형 (클래스 없음) — user 객체 받아 mode 자동 선택 |
| 모드 관리 | paper/real 둘 다 등록 가능, `PUT /auth/mode`로 전환 |
| 주문 체결 확인 | Celery 비동기 폴링 (10초 간격, 최대 5회) + 체결 이메일 |
| 포트폴리오 잔고 | DB 자체 관리 (앱 주문만 추적), 현재가는 pykrx 조회 |
| 리스크 | 종목별 한도 + 일일 손실 한도, enforce_hard_stop 설정으로 차단/경고 선택 |
| 알림 이메일 | 가입 이메일 기본, `alert_settings.notification_email`로 변경 가능 |

---

## 2. 파일 목록

### 신규 생성
| 파일 | 역할 |
|---|---|
| `backend/services/risk_service.py` | 종목별/일일 손실 한도 체크 |
| `backend/tasks/email_tasks.py` | 체결/리스크/가격 이메일 Celery 태스크 |
| `backend/tasks/order_tasks.py` | 체결 폴링 Celery 태스크 |
| `backend/api/routes/trades.py` | 주문 실행/목록/취소 라우터 |
| `backend/api/routes/portfolio.py` | 포트폴리오 현황/성과/지표/CSV 라우터 |
| `backend/api/routes/risk.py` | 리스크 설정 조회/수정 라우터 |
| `backend/api/routes/alerts.py` | 알림 설정 조회/수정 라우터 |
| `tests/test_trades.py` | 주문 API 통합 테스트 |
| `tests/test_portfolio.py` | 포트폴리오 API 통합 테스트 |
| `tests/test_risk.py` | risk_service 유닛 + API 테스트 |

### 수정
| 파일 | 변경 내용 |
|---|---|
| `backend/services/kis_service.py` | 완전 재작성 — 주문/취소/잔고/체결조회 함수 추가 |
| `backend/models/risk.py` | `RiskSettings`에 `enforce_hard_stop` 컬럼 추가 |
| `backend/models/risk.py` | `AlertSettings`에 `notification_email` 컬럼 추가 |
| `backend/api/routes/auth.py` | `PUT /auth/mode` 엔드포인트 추가 |
| `backend/tasks/__init__.py` | `include` 목록에 `tasks.email_tasks`, `tasks.order_tasks` 추가 |
| `backend/main.py` | APScheduler 추가 스케줄 등록 |
| `db/migrations/` | risk_settings/alert_settings 컬럼 추가 마이그레이션 |
| `docs/progress.md` | Phase 4-A 완료 업데이트 |

---

## 3. KIS 서비스 (`backend/services/kis_service.py`)

완전 재작성. 기존 `test_kis_connection` 함수 유지 + 주문/조회 함수 추가.

### 3.1 URL/TR ID 매핑

```python
_REAL_URL   = "https://openapi.koreainvestment.com:9443"
_PAPER_URL  = "https://openapivts.koreainvestment.com:29443"

_TR_IDS = {
    "buy":     {"real": "TTTC0802U", "paper": "VTTC0802U"},
    "sell":    {"real": "TTTC0801U", "paper": "VTTC0801U"},
    "cancel":  {"real": "TTTC0803U", "paper": "VTTC0803U"},
    "balance": {"real": "TTTC8434R", "paper": "VTTC8434R"},
    "fill":    {"real": "TTTC8001R", "paper": "VTTC8001R"},
}
```

### 3.2 내부 헬퍼

```python
def _get_base_url(mode: str) -> str: ...
def _get_tr_id(action: str, mode: str) -> str: ...
def _get_keys(user) -> tuple[str, str, str]:
    # mode에 따라 paper or real 키 복호화 반환
    # (app_key, app_secret, account_no)
```

### 3.3 공개 함수

```python
async def place_order(user, stock_code: str, order_type: str,
                      price_type: str, quantity: int,
                      price: int = 0) -> dict:
    """
    KIS 주문 실행.
    반환: {kis_order_no, status}
    실패 시 KISAPIError 발생
    """

async def cancel_order(user, kis_order_no: str) -> dict:
    """미체결 주문 취소."""

async def poll_fill(user, kis_order_no: str) -> dict | None:
    """
    체결 확인.
    반환: {executed_price, filled_qty, filled_at} or None (미체결)
    """

async def get_balance(user) -> dict:
    """예수금 조회. 반환: {cash, total_eval}"""

async def get_holdings(user) -> list[dict]:
    """실제 보유종목 조회 (포트폴리오 동기화용)."""
```

---

## 4. 리스크 서비스 (`backend/services/risk_service.py`)

```python
class RiskLimitExceeded(Exception):
    def __init__(self, reason: str, warning_only: bool = False): ...

async def check_order(
    user, stock_code: str, quantity: int, price: int, db: AsyncSession
) -> str | None:
    """
    주문 전 리스크 체크.
    반환: None (통과) or warning 메시지 (경고 모드)
    한도 초과 + enforce_hard_stop=True → RiskLimitExceeded 발생 (400)
    한도 초과 + enforce_hard_stop=False → warning 메시지 반환

    체크 항목:
    1. 종목별 한도: (기존 보유금액 + 신규 주문금액) / 포트폴리오 총액 > max_per_stock_pct
    2. 일일 손실 한도: 오늘 실현손실 / 어제 포트폴리오 총액 > daily_loss_limit_pct
    """

async def get_or_create_settings(user_id: UUID, db: AsyncSession) -> RiskSettings:
    """없으면 기본값으로 생성."""
```

---

## 5. 주문 실행 흐름

### 5.1 POST /trades/order

```
요청: {stock_code, order_type(BUY/SELL), price_type(MARKET/LIMIT), quantity, price}

① JWT → user 조회
② user.mode == "demo" → 403 ("KIS 키를 먼저 등록하세요")
③ risk_service.check_order() → 한도 초과 시 처리
④ kis_service.place_order(user, ...) → {kis_order_no}
⑤ trades INSERT:
     status=PENDING, mode=user.mode, kis_order_no, ai_signal_at_order
⑥ order_tasks.poll_order_fill.delay(trade_id, user_id, kis_order_no, user.mode)
⑦ 응답: {trade_id, status: "PENDING", warning?: "..."}
```

### 5.2 Celery 체결 폴링 (`order_tasks.poll_order_fill`)

```python
@celery_app.task(bind=True, max_retries=5)
def poll_order_fill(self, trade_id, user_id, kis_order_no, mode):
    result = asyncio.run(_poll_async(...))
    if result == "filled":
        return
    if self.request.retries < 5:
        raise self.retry(countdown=10)
    # 5회 후 미체결: 그대로 PENDING 유지
```

체결 확인 시:
- `trades`: `status=FILLED`, `executed_price`, `filled_at` 업데이트
- `portfolios` UPSERT: `quantity` 증감, `avg_price` 가중평균 재계산
- `email_tasks.send_fill_notification.delay(user_id, trade_id)`

---

## 6. 포트폴리오 관리

### GET /portfolio 응답 계산

```python
# DB에서 보유종목 조회 (현재 mode 기준)
holdings = await db.query(Portfolio).filter(user_id, mode)

# pykrx로 현재가 조회 (Redis TTL 30초)
for holding in holdings:
    current_price = await get_stock_current_price(holding.stock_code)
    eval_amount = holding.quantity * current_price
    profit_loss = eval_amount - (holding.quantity * holding.avg_price)
    return_pct = profit_loss / (holding.quantity * holding.avg_price) * 100
```

### GET /portfolio/metrics

```python
# trades 히스토리에서 계산
# MDD: max(cummax - current) / cummax
# 샤프비율: (mean_return - 0) / std_return * sqrt(252)
# 승률: FILLED trades 중 profit > 0 비율
```

### GET /portfolio/export

`StreamingResponse` + CSV 형식, 컬럼: 종목코드, 종목명, 수량, 평균매수가, 현재가, 평가금액, 수익률

---

## 7. 이메일 태스크 (`backend/tasks/email_tasks.py`)

```python
@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_fill_notification(self, user_id, trade_id):
    """체결 완료 이메일 (notification_email or user.email)."""

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_risk_alert(self, user_id, reason):
    """리스크 한도 초과 이메일."""

@celery_app.task
def check_price_alerts():
    """관심종목 목표가 도달 시 이메일 — APScheduler 5분 간격 호출."""

@celery_app.task
def check_daily_loss():
    """일일 손실 한도 체크 — APScheduler 10분 간격 호출."""
```

이메일 발신: `email_service.py`의 기존 SendGrid 래퍼 재사용.
수신 주소: `alert_settings.notification_email` (없으면 `user.email` fallback).

---

## 8. API 엔드포인트 명세

### 거래 (`/trades`)
| Method | Path | 인증 | 설명 |
|---|---|---|---|
| POST | `/trades/order` | ✅ | 주문 실행 (리스크 체크 포함) |
| GET | `/trades` | ✅ | 주문 목록 (`?status=PENDING&mode=paper`) |
| DELETE | `/trades/{id}` | ✅ | 미체결 주문 취소 |

### 포트폴리오 (`/portfolio`)
| Method | Path | 인증 | 설명 |
|---|---|---|---|
| GET | `/portfolio` | ✅ | 보유종목 현황 + 수익률 |
| GET | `/portfolio/performance` | ✅ | 일별 평가액 히스토리 |
| GET | `/portfolio/metrics` | ✅ | MDD, 샤프비율, 승률 |
| GET | `/portfolio/export` | ✅ | CSV 스트리밍 다운로드 |

### 리스크 (`/risk`)
| Method | Path | 인증 | 설명 |
|---|---|---|---|
| GET | `/risk/settings` | ✅ | 리스크 설정 조회 |
| PUT | `/risk/settings` | ✅ | 리스크 설정 수정 |

### 알림 (`/alerts`)
| Method | Path | 인증 | 설명 |
|---|---|---|---|
| GET | `/alerts/settings` | ✅ | 알림 설정 조회 |
| PUT | `/alerts/settings` | ✅ | 알림 설정 + notification_email 수정 |

### 인증 (`/auth`) 추가
| Method | Path | 인증 | 설명 |
|---|---|---|---|
| PUT | `/auth/mode` | ✅ | paper ↔ real 모드 전환 |

---

## 9. DB 스키마 변경

### risk_settings 컬럼 추가
```sql
ALTER TABLE risk_settings ADD COLUMN enforce_hard_stop BOOLEAN DEFAULT TRUE;
```

### alert_settings 컬럼 추가
```sql
ALTER TABLE alert_settings ADD COLUMN notification_email VARCHAR(255);
-- NULL이면 user.email 사용
```

→ Alembic 버전 8 (v8): 두 컬럼 추가 마이그레이션 (v7은 복합 인덱스로 사용됨)

---

## 10. 테스트 전략

| 파일 | 테스트 수 (목표) | 내용 |
|---|---|---|
| `tests/test_trades.py` | 8 | demo 모드 차단, 리스크 차단, PENDING 응답, 취소, 목록 조회 |
| `tests/test_portfolio.py` | 5 | 빈 포트폴리오, 수익률 계산 (현재가 mock), CSV 다운로드 |
| `tests/test_risk.py` | 6 | 종목별 한도 초과, 일일 손실 한도, 경고 모드, 설정 CRUD |
| **합계** | **19** | KIS API, pykrx, Celery 모두 mock |

**Celery 테스트:** `CELERY_TASK_ALWAYS_EAGER=True` 설정으로 동기 실행.
**KIS mock:** `unittest.mock.AsyncMock`으로 `kis_service` 함수 전체 대체.

---

## 11. 구현 순서

1. DB 마이그레이션 (v7: risk_settings + alert_settings 컬럼 추가)
2. `kis_service.py` 완전 재작성
3. `risk_service.py` 구현
4. `tasks/order_tasks.py` 구현
5. `tasks/email_tasks.py` 구현
6. `api/routes/trades.py` 구현
7. `api/routes/portfolio.py` 구현
8. `api/routes/risk.py` + `api/routes/alerts.py` 구현
9. `auth.py` PUT /auth/mode 추가
10. `tasks/__init__.py` + `main.py` 업데이트
11. 테스트 작성
12. `docs/progress.md` 업데이트
