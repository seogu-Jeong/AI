# Phase 4-C — 투자 시뮬레이터 설계

**작성일:** 2026-06-04
**담당:** hygrenn (백엔드)
**레퍼런스:** TRD 섹션 5.7, 15-A

---

## 1. 결정 사항

| 항목 | 결정 |
|---|---|
| 데이터 소스 | pykrx (기존 의존성 재사용) |
| 종목 범위 | 한국 주식 고정 리스트 80개 (KOSPI 대형주) |
| 가격 캐시 | PostgreSQL `price_cache` 테이블 (영구 저장) |
| 미장 지원 | v2로 연기 (market 컬럼은 지금 추가해 스키마 준비) |
| 실행 방식 | 동기 계산 (lumpsum/recurring), SSE 스트리밍 (download) |

---

## 2. 신규/수정 파일

| 파일 | 역할 |
|---|---|
| `db/migrations/versions/e6f7a2b3c4d5_add_price_cache.py` | price_cache 테이블 생성 |
| `backend/models/price_cache.py` | SQLAlchemy 모델 |
| `backend/services/simulator_service.py` | 계산 엔진 + 데이터 로더 |
| `backend/api/routes/simulate.py` | 4개 엔드포인트 |
| `tests/test_simulate.py` | 통합 5개 + 단위 3개 |
| `backend/main.py` | simulate 라우터 등록 |
| `docs/progress.md` | Phase 4-C 완료 표기 |

---

## 3. DB 스키마

```sql
CREATE TABLE price_cache (
    ticker      VARCHAR(20)   NOT NULL,
    trade_date  DATE          NOT NULL,
    close_price NUMERIC(12,2) NOT NULL,
    market      VARCHAR(10)   NOT NULL DEFAULT 'KR',
    PRIMARY KEY (ticker, trade_date, market)
);
CREATE INDEX idx_price_cache_ticker_market ON price_cache (ticker, market);
```

- `market` 컬럼은 v2 미장 지원을 위해 지금 추가. 현재는 'KR' 고정.
- 결측치(휴장일)는 저장하지 않고 서비스 레이어에서 ffill 처리.

---

## 4. 고정 종목 리스트 (`SIMULATOR_TICKERS`)

`simulator_service.py` 상단 상수. `{"code": str, "name": str}` 형태 80개.

KOSPI 대형주 기준 예시 (일부):
```python
SIMULATOR_TICKERS = [
    {"code": "005930", "name": "삼성전자"},
    {"code": "000660", "name": "SK하이닉스"},
    {"code": "373220", "name": "LG에너지솔루션"},
    {"code": "207940", "name": "삼성바이오로직스"},
    {"code": "005380", "name": "현대차"},
    # ... 75개 추가
]
```

---

## 5. 서비스 레이어 (`simulator_service.py`)

### 5-1. `get_prices(ticker, start, end, db) → dict[str, float]`

```
1. price_cache에서 (ticker, market='KR', trade_date BETWEEN start AND end) 조회
2. 요청 기간 중 누락된 날짜가 있으면:
   a. pykrx.stock.get_market_ohlcv_by_date(start, end, ticker)를 asyncio.to_thread로 호출
   b. 조회된 종가를 price_cache에 UPSERT
3. 전체 날짜 시리즈에 ffill 적용 (휴장일 → 직전 영업일 가격)
4. 반환: {"2022-01-03": 76000.0, "2022-01-04": 75500.0, ...}
```

### 5-2. `calc_lumpsum(ticker, buy_date, sell_date, amount_krw, db) → dict`

```
1. prices = await get_prices(ticker, buy_date, sell_date, db)
2. buy_date_actual = 가격 있는 가장 빠른 날짜 (≥ buy_date)
3. sell_date_actual = 가격 있는 가장 늦은 날짜 (≤ sell_date)
4. shares = int(amount_krw / buy_price)
5. buy_value = shares * buy_price
6. sell_value = shares * sell_price
7. profit_krw = sell_value - buy_value
8. return_pct = profit_krw / buy_value * 100
9. chart_data = [{"date": d, "return_pct": (p/buy_price - 1)*100} for d, p in prices]
```

### 5-3. `calc_recurring(ticker, start_date, end_date, monthly_krw, db) → dict`

```
1. prices = await get_prices(ticker, start_date, end_date, db)
2. trading_days = 매월 첫 번째 영업일 리스트 (start~end)
3. 매 trading_day마다:
   - shares_this_month = int(monthly_krw / price)
   - total_shares += shares_this_month
   - total_invested += shares_this_month * price
4. final_value = total_shares * prices[end_date_actual]
5. avg_buy_price = total_invested / total_shares
6. return_pct = (final_value - total_invested) / total_invested * 100
7. chart_data = 누적 투자액 vs 평가액 시리즈
```

### 5-4. `download_tickers(db)` — async generator

```
for i, t in enumerate(SIMULATOR_TICKERS):
    await get_prices(t["code"], date.today().replace(year=date.today().year - 5), date.today(), db)
    yield {"current": i+1, "total": len(SIMULATOR_TICKERS),
           "ticker": t["code"], "name": t["name"]}
```

---

## 6. API 엔드포인트 (`routes/simulate.py`)

### POST /simulate/lumpsum

**요청:**
```json
{
  "tickers": ["005930", "000660"],
  "buy_date": "2022-01-03",
  "sell_date": "2026-05-31",
  "amount_krw": 1000000
}
```
검증: `sell_date > buy_date`, `amount_krw > 0`, `tickers` 1~10개, 종목코드 6자리

**응답:**
```json
{
  "buy_date_actual": "2022-01-03",
  "sell_date_actual": "2026-05-30",
  "results": [
    {
      "ticker": "005930",
      "name": "삼성전자",
      "shares": 13,
      "buy_price": 76000,
      "sell_price": 73400,
      "buy_value_krw": 988000,
      "sell_value_krw": 954200,
      "profit_krw": -33800,
      "return_pct": -3.42,
      "chart_data": [{"date": "2022-01-03", "return_pct": 0.0}, "..."]
    }
  ]
}
```

### POST /simulate/recurring

**요청:**
```json
{
  "tickers": ["005930"],
  "start_date": "2020-01-02",
  "end_date": "2026-05-31",
  "monthly_amount_krw": 300000
}
```
검증: `end_date > start_date`, `monthly_amount_krw > 0`, `tickers` 1~5개

**응답:**
```json
{
  "results": [{
    "ticker": "005930",
    "name": "삼성전자",
    "total_invested_krw": 19200000,
    "total_shares": 252,
    "avg_buy_price": 76190,
    "current_value_krw": 18496800,
    "return_pct": -3.56,
    "total_purchases": 64,
    "chart_data": [
      {"date": "2020-01-02", "invested": 300000, "value": 300000}
    ]
  }]
}
```

### GET /simulate/data-status

**응답:**
```json
{"ready": true, "ticker_count": 80, "last_updated": "2026-06-04"}
```
- `ready`: price_cache에 SIMULATOR_TICKERS 전체 데이터가 존재하면 true
- `last_updated`: price_cache의 가장 최근 trade_date

### GET /simulate/download (SSE)

- `Content-Type: text/event-stream`
- `download_tickers()` generator를 sse_starlette `EventSourceResponse`로 스트리밍
- 인증 필요 (get_current_user)

```
event: progress
data: {"current": 1, "total": 80, "ticker": "005930", "name": "삼성전자"}

event: progress
data: {"current": 2, "total": 80, "ticker": "000660", "name": "SK하이닉스"}

event: complete
data: {"message": "다운로드 완료", "total": 80}
```

---

## 7. 테스트 (`tests/test_simulate.py`)

### 통합 테스트 (5개)

| 테스트 | 검증 |
|---|---|
| `test_lumpsum_returns_result` | `simulator_service` mock → 200 + 필수 필드 |
| `test_lumpsum_invalid_dates` | sell_date ≤ buy_date → 422 |
| `test_recurring_returns_result` | `simulator_service` mock → 200 + chart_data |
| `test_data_status_not_ready` | 빈 price_cache → `{"ready": false}` |
| `test_download_sse_content_type` | SSE Content-Type 확인 |

### 단위 테스트 (3개)

| 테스트 | 검증 |
|---|---|
| `test_calc_lumpsum_logic` | 고정 가격 dict → shares/profit_krw/return_pct 정확성 |
| `test_calc_recurring_logic` | 3개월 적립 → total_invested/avg_buy_price 정확성 |
| `test_next_trading_day` | 토요일 입력 → 다음 월요일 반환 |

---

## 8. 에러 처리

| 상황 | 응답 |
|---|---|
| pykrx 조회 실패 | 502 + 메시지 |
| 해당 기간 데이터 없음 | 404 |
| 매수/매도일 모두 휴장 | 422 + "거래일이 없습니다" |
| tickers 빈 배열 | 422 (Pydantic) |
