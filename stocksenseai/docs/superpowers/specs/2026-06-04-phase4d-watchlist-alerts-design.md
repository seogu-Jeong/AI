# Phase 4-D — 관심종목 + 알림 설계

**작성일:** 2026-06-04
**담당:** hygrenn (백엔드)
**레퍼런스:** TRD 섹션 4.5, 9.1, 9.2

---

## 1. 결정 사항

| 항목 | 결정 |
|---|---|
| 그룹 관리 | 풀 CRUD (생성/수정/삭제 + 아이템 관리) |
| 현재가 조회 | Redis 캐시 우선 (market_service.get_current_price 재사용) |
| 일일 손실 초과 시 | trading_blocked 자동 설정 + 이메일 발송 |
| 알림 재발송 방지 | Redis TTL 24시간 쿨다운 키 |

---

## 2. 기존 인프라 (수정 불필요)

| 항목 | 파일 | 상태 |
|---|---|---|
| WatchlistGroup/Item 모델 | `backend/models/watchlist.py` | ✅ 완성 |
| watchlist 마이그레이션 | `db/migrations/versions/b2c3d4e5f6a1_add_watchlists.py` | ✅ 완성 |
| AlertSettings 모델 + API | `backend/models/risk.py`, `backend/api/routes/alerts.py` | ✅ 완성 |
| APScheduler 스케줄 등록 | `backend/main.py` | ✅ 완성 |
| send_risk_alert Celery 태스크 | `backend/tasks/email_tasks.py` | ✅ 완성 |
| trading_blocked 체크 | `backend/services/risk_service.py` | ✅ 완성 |

---

## 3. 신규/수정 파일

| 파일 | 역할 |
|---|---|
| `backend/api/routes/watchlist.py` | 관심종목 CRUD 라우터 8개 (신규) |
| `backend/tasks/email_tasks.py` | check_price_alerts, check_daily_loss 구현 (수정) |
| `tests/test_watchlist.py` | 관심종목 통합 테스트 6개 (신규) |
| `tests/test_alert_tasks.py` | 알림 태스크 단위 테스트 5개 (신규) |
| `backend/main.py` | watchlist 라우터 등록 (수정) |
| `docs/progress.md` | Phase 4-D 완료 표기 (수정) |

---

## 4. 관심종목 API (`routes/watchlist.py`)

### 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| `GET` | `/watchlist/groups` | 내 그룹 목록 (아이템 포함) |
| `POST` | `/watchlist/groups` | 그룹 생성 |
| `PUT` | `/watchlist/groups/{group_id}` | 그룹명/순서 수정 |
| `DELETE` | `/watchlist/groups/{group_id}` | 그룹 삭제 (아이템 CASCADE) |
| `GET` | `/watchlist/items` | 전체 관심종목 평면 리스트 |
| `POST` | `/watchlist/items` | 종목 추가 |
| `PUT` | `/watchlist/items/{item_id}` | 목표가/순서 수정 |
| `DELETE` | `/watchlist/items/{item_id}` | 종목 삭제 |

### Pydantic 모델

```python
class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    sort_order: int = 0

class GroupUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=50)
    sort_order: int | None = None

class ItemCreate(BaseModel):
    group_id: str                      # UUID
    stock_code: str                    # 6자리 숫자 검증
    stock_name: str | None = None
    target_price_high: float | None = Field(default=None, gt=0)
    target_price_low: float | None = Field(default=None, gt=0)
    sort_order: int = 0

class ItemUpdate(BaseModel):
    target_price_high: float | None = Field(default=None, gt=0)
    target_price_low: float | None = Field(default=None, gt=0)
    sort_order: int | None = None
    group_id: str | None = None        # 그룹 이동
```

### GET /watchlist/groups 응답

```json
[
  {
    "id": "uuid",
    "name": "주목 종목",
    "sort_order": 0,
    "created_at": "2026-06-04T00:00:00+09:00",
    "items": [
      {
        "id": "uuid",
        "stock_code": "005930",
        "stock_name": "삼성전자",
        "target_price_high": 80000,
        "target_price_low": 65000,
        "sort_order": 0
      }
    ]
  }
]
```

### 에러 처리

| 상황 | 응답 |
|---|---|
| 다른 유저의 그룹/아이템 접근 | 404 |
| 동일 종목 중복 추가 | 409 |
| 존재하지 않는 group_id로 아이템 추가 | 404 |
| stock_code 6자리 숫자 아님 | 422 |

---

## 5. 알림 태스크 (`tasks/email_tasks.py`)

### `check_price_alerts()` — 5분마다

```
1. DB에서 target_price_high 또는 target_price_low가 설정된
   watchlist_items 전체 조회 (user 조인 포함)

2. 유저별 alert_settings.watchlist_price 확인
   → False면 해당 유저 아이템 전체 skip

3. 종목별 현재가 조회:
   market_service.get_stock_current_price(stock_code)
   → Redis 캐시 우선, 미스 시 pykrx

4. 조건 체크:
   - 현재가 >= target_price_high → alert_type = "high"
   - 현재가 <= target_price_low  → alert_type = "low"

5. Redis key 확인: "price_alert:{user_id}:{stock_code}:{alert_type}"
   - 존재하면 → skip (24시간 쿨다운)
   - 없으면  → send_price_alert_email.delay() 발송
              → Redis key SET EX 86400

6. 장 시간(09:00~15:30 KST) 외에는 실행해도 가격이 안 바뀌므로
   실질적 중복 발송은 Redis 쿨다운으로 방지
```

### `check_daily_loss()` — 10분마다

```
1. 오늘 SELL FILLED 거래가 있는 user_id 목록 조회

2. 유저별:
   a. alert_settings.daily_loss_limit 확인 → False면 skip
   b. risk_settings 조회 (get_or_create_settings)
   c. trading_blocked 이미 True면 skip
   d. _get_today_loss(user_id, mode, db) 호출
   e. _get_portfolio_total(user_id, mode, db) 호출
   f. loss_pct = today_loss / portfolio_total * 100
   g. loss_pct > daily_loss_limit_pct 이면:
      - risk_settings.trading_blocked = True, blocked_at = now()
      - Redis key "daily_loss_alert:{user_id}" TTL 24h (중복 발송 방지)
      - send_risk_alert.delay(user_id, reason) 발송
```

### 신규 Celery 태스크: `send_price_alert_email`

```python
@celery_app.task(bind=True, max_retries=3)
def send_price_alert_email(self, user_id: str, stock_code: str,
                           stock_name: str, current_price: float,
                           target_price: float, alert_type: str) -> None:
    """목표가/손절가 도달 이메일. alert_type: 'high' | 'low'"""
```

---

## 6. 테스트

### `tests/test_watchlist.py` — 통합 6개 (client fixture 사용)

| 테스트 | 검증 |
|---|---|
| `test_get_groups_empty` | 신규 유저 → `[]` 반환 |
| `test_create_group` | POST → 201, name/id 반환 |
| `test_add_item_to_group` | POST item → GET groups에 포함됨 |
| `test_add_duplicate_item` | 동일 stock_code 재추가 → 409 |
| `test_update_item_target_price` | PUT target_price_high → 200, 값 반영 |
| `test_delete_group_cascades` | DELETE group → items도 삭제 |

### `tests/test_alert_tasks.py` — 단위 5개 (mock 기반)

| 테스트 | 검증 |
|---|---|
| `test_price_alert_high_triggered` | 현재가 ≥ 목표가 → send_price_alert_email.delay 호출 |
| `test_price_alert_cooldown_skip` | Redis 쿨다운 키 존재 → delay 미호출 |
| `test_price_alert_setting_disabled` | watchlist_price=False → delay 미호출 |
| `test_daily_loss_blocks_trading` | loss_pct > limit → trading_blocked=True + send_risk_alert.delay 호출 |
| `test_daily_loss_within_limit` | loss_pct ≤ limit → 차단 없음 |
