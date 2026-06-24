# Phase 2: 실시간 시세 + WebSocket Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** KIS WebSocket Pool로 실시간 체결/호가를 Redis Pub/Sub에 발행하고, SSE 엔드포인트로 클라이언트에 스트리밍하며, KIS REST로 분봉 차트·호가·체결 데이터를 제공한다.

**Architecture:** system-level KIS 키(`SYSTEM_KIS_*` settings)로 shared market-data 레이어를 구성한다. KIS REST는 orderbook/trades/intraday OHLCV를 Redis에 캐시하고, KIS WebSocket Pool은 구독 종목을 41개씩 배치로 묶어 세션을 관리하며 수신 메시지를 `stock:{code}` Redis 채널에 발행한다. FastAPI SSE 엔드포인트는 해당 채널을 구독해 클라이언트로 스트리밍한다.

**Tech Stack:** FastAPI, `sse-starlette==2.1.3`, `websockets==12.0`, `httpx`, `redis[asyncio]`, `pykrx`(일봉 유지), Python `zoneinfo`

---

## 파일 구조

| 역할 | 파일 |
|---|---|
| KIS OAuth 토큰 캐시 | `backend/services/kis_token_service.py` (신규) |
| KIS REST 시세 (호가·체결·분봉) | `backend/services/kis_market_service.py` (신규) |
| KIS WebSocket Pool | `backend/services/websocket_service.py` (신규) |
| SSE 엔드포인트 | `backend/api/routes/realtime.py` (신규) |
| stocks 라우트 확장 | `backend/api/routes/stocks.py` (수정) |
| settings 확장 | `backend/core/config.py` (수정) |
| lifespan + 라우터 등록 | `backend/main.py` (수정) |
| 의존성 추가 | `backend/requirements.txt` (수정) |
| Phase 2 테스트 | `tests/test_phase2.py` (신규) |

---

## 사전 지식: KIS OpenAPI

### KIS BASE URL
```python
KIS_REAL = "https://openapi.koreainvestment.com:9443"
KIS_PAPER = "https://openapivts.koreainvestment.com:29443"
KIS_WS_REAL = "ws://ops.koreainvestment.com:21000/"
KIS_WS_PAPER = "ws://ops.koreainvestment.com:31000/"
```

### 사용하는 TR ID
| TR ID | 설명 | 프로토콜 |
|---|---|---|
| `FHKST01010200` | 10단 호가 | REST GET |
| `FHKST01010300` | 최근 체결 | REST GET |
| `FHKST03010200` | 분봉 OHLCV | REST GET |
| `H0STCNT0` | 실시간 체결 | WebSocket |
| `H0STASP0` | 실시간 호가 | WebSocket |

### access_token (REST 호출용)
```
POST {base}/oauth2/tokenP
Content-Type: application/json
{"grant_type":"client_credentials","appkey":"...","appsecret":"..."}
→ {"access_token":"...","expires_in":86400}
```

### approval_key (WebSocket 연결용)
```
POST {base}/oauth2/Approval
Content-Type: application/json
{"grant_type":"client_credentials","appkey":"...","secretkey":"..."}
→ {"approval_key":"..."}
```

### KIS REST 공통 헤더 (호가·체결·분봉)
```
Authorization: Bearer {access_token}
appkey: {SYSTEM_KIS_APP_KEY}
appsecret: {SYSTEM_KIS_APP_SECRET}
tr_id: {TR_ID}
custtype: P
Content-Type: application/json; charset=utf-8
```

### WebSocket 구독 메시지 (JSON → WS send)
```json
{
  "header": {
    "approval_key": "...",
    "custtype": "P",
    "tr_type": "1",
    "content-type": "utf-8"
  },
  "body": {"input": {"tr_id": "H0STCNT0", "tr_key": "005930"}}
}
```

### WebSocket 수신 포맷 (pipe + caret)
```
0|H0STCNT0|001|005930^093000^60100^2^100^0.17^...^1000^...
         ^-- TR_ID  ^-- COUNT ^-- data fields separated by ^
```
- 첫 문자 `0` = 일반 데이터, `1` = PINGPONG

H0STCNT0 key field positions (0-indexed, `^` split):
- 0: 종목코드
- 1: 체결시간(HHMMSS)
- 2: 현재가
- 3: 전일대비부호 (1=상한/2=상승/3=보합/4=하한/5=하락)
- 4: 전일대비
- 5: 전일대비율
- 12: 체결거래량

---

## Task 1: KIS OAuth Token Service

**Files:**
- Create: `backend/services/kis_token_service.py`
- Test: `tests/test_phase2.py` (첫 번째 섹션)

- [ ] **Step 1: test_phase2.py 파일 생성 및 token service 테스트 작성**

```python
# tests/test_phase2.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ─── Task 1: KIS Token Service ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_access_token_fetches_and_caches():
    """access_token을 KIS에서 받아 Redis에 캐시한다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None  # 캐시 miss

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"access_token": "tok_abc", "expires_in": 86400}

    with patch("services.kis_token_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=fake_resp)
        mock_client_cls.return_value = mock_client

        from services.kis_token_service import get_access_token
        token = await get_access_token("key", "secret", "paper")

    assert token == "tok_abc"
    mock_redis.setex.assert_called_once()
    args = mock_redis.setex.call_args[0]
    assert "access_token" in args[0]
    assert args[1] == 86340  # expires_in - 60


@pytest.mark.asyncio
async def test_get_access_token_uses_cache():
    """캐시된 access_token이 있으면 KIS를 호출하지 않는다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = b"tok_cached"

    with patch("services.kis_token_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_client_cls:
        from services.kis_token_service import get_access_token
        token = await get_access_token("key", "secret", "paper")

    assert token == "tok_cached"
    mock_client_cls.assert_not_called()


@pytest.mark.asyncio
async def test_get_approval_key_fetches_and_caches():
    """approval_key를 KIS에서 받아 Redis에 82800초로 캐시한다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"approval_key": "approv_xyz"}

    with patch("services.kis_token_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=fake_resp)
        mock_client_cls.return_value = mock_client

        from services.kis_token_service import get_approval_key
        key = await get_approval_key("key", "secret", "paper")

    assert key == "approv_xyz"
    args = mock_redis.setex.call_args[0]
    assert args[1] == 82800
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /path/to/FinalProject && python3 -m pytest tests/test_phase2.py -v 2>&1 | tail -15
```
Expected: `ImportError` or `ModuleNotFoundError: No module named 'services.kis_token_service'`

- [ ] **Step 3: kis_token_service.py 구현**

```python
# backend/services/kis_token_service.py
import httpx

from core.redis_client import get_redis

_KIS_REAL = "https://openapi.koreainvestment.com:9443"
_KIS_PAPER = "https://openapivts.koreainvestment.com:29443"


def _base(mode: str) -> str:
    return _KIS_PAPER if mode == "paper" else _KIS_REAL


async def get_access_token(app_key: str, app_secret: str, mode: str) -> str:
    redis = await get_redis()
    cache_key = f"access_token:{mode}:{app_key[:8]}"

    cached = await redis.get(cache_key)
    if cached:
        return cached.decode() if isinstance(cached, bytes) else cached

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_base(mode)}/oauth2/tokenP",
            json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    token: str = data["access_token"]
    ttl: int = int(data.get("expires_in", 86400)) - 60
    await redis.setex(cache_key, ttl, token)
    return token


async def get_approval_key(app_key: str, app_secret: str, mode: str) -> str:
    redis = await get_redis()
    cache_key = f"approval_key:{mode}:{app_key[:8]}"

    cached = await redis.get(cache_key)
    if cached:
        return cached.decode() if isinstance(cached, bytes) else cached

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"{_base(mode)}/oauth2/Approval",
            json={"grant_type": "client_credentials", "appkey": app_key, "secretkey": app_secret},
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()

    key: str = data["approval_key"]
    await redis.setex(cache_key, 82800, key)
    return key
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_phase2.py -v 2>&1 | tail -15
```
Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
git add backend/services/kis_token_service.py tests/test_phase2.py
git commit -m "feat: KIS OAuth token service (access_token + approval_key with Redis cache)"
```

---

## Task 2: KIS REST Market Service (호가·체결·분봉)

**Files:**
- Create: `backend/services/kis_market_service.py`
- Test: `tests/test_phase2.py` (Task 2 섹션 추가)

**Note:** 이 서비스는 `settings.SYSTEM_KIS_APP_KEY` 등 system-level 키를 사용한다.  
키가 비어 있으면 빈 결과를 반환하며 에러를 발생시키지 않는다.

- [ ] **Step 1: config.py에 SYSTEM_KIS 설정 추가**

`backend/core/config.py` — `REFRESH_TOKEN_EXPIRE_DAYS` 아래에 추가:

```python
    SYSTEM_KIS_APP_KEY: str = ""
    SYSTEM_KIS_APP_SECRET: str = ""
    SYSTEM_KIS_MODE: str = "paper"
```

- [ ] **Step 2: 테스트 추가 (tests/test_phase2.py 하단에 append)**

```python
# ─── Task 2: KIS REST Market Service ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_orderbook_returns_empty_when_no_system_key():
    """SYSTEM_KIS_APP_KEY가 없으면 빈 호가를 반환한다."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        from services.kis_market_service import get_orderbook
        result = await get_orderbook("005930")
    assert result == {"code": "005930", "asks": [], "bids": []}


@pytest.mark.asyncio
async def test_get_orderbook_parses_kis_response():
    """KIS 응답을 10단 호가 형태로 파싱한다."""
    # KIS 응답 output1 구조: 매도호가/수량 10쌍 + 매수호가/수량 10쌍
    fake_output1 = {
        "askp1": "60100", "askp_rsqn1": "500",
        "askp2": "60200", "askp_rsqn2": "300",
        **{f"askp{i}": str(60100 + (i-1)*100) for i in range(3, 11)},
        **{f"askp_rsqn{i}": "100" for i in range(3, 11)},
        "bidp1": "60000", "bidp_rsqn1": "1000",
        "bidp2": "59900", "bidp_rsqn2": "800",
        **{f"bidp{i}": str(60000 - (i-1)*100) for i in range(3, 11)},
        **{f"bidp_rsqn{i}": "200" for i in range(3, 11)},
    }
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"output1": fake_output1}

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("services.kis_market_service.settings") as mock_settings, \
         patch("services.kis_market_service.get_access_token", return_value="tok"), \
         patch("services.kis_market_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_cls:
        mock_settings.SYSTEM_KIS_APP_KEY = "testkey"
        mock_settings.SYSTEM_KIS_APP_SECRET = "testsecret"
        mock_settings.SYSTEM_KIS_MODE = "paper"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=fake_resp)
        mock_cls.return_value = mock_client

        from services.kis_market_service import get_orderbook
        result = await get_orderbook("005930")

    assert len(result["asks"]) == 10
    assert len(result["bids"]) == 10
    assert result["asks"][0] == {"price": 60100, "qty": 500}
    assert result["bids"][0] == {"price": 60000, "qty": 1000}


@pytest.mark.asyncio
async def test_get_intraday_ohlcv_returns_empty_when_no_system_key():
    """SYSTEM_KIS_APP_KEY가 없으면 빈 분봉 데이터를 반환한다."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        from services.kis_market_service import get_intraday_ohlcv
        result = await get_intraday_ohlcv("005930", "1min")
    assert result == []
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_phase2.py::test_get_orderbook_returns_empty_when_no_system_key -v
```
Expected: `ImportError` — kis_market_service 없음

- [ ] **Step 4: kis_market_service.py 구현**

```python
# backend/services/kis_market_service.py
import json

import httpx

from core.config import settings
from core.redis_client import get_redis
from services.kis_token_service import get_access_token
from services.market_service import _is_market_open


def _base(mode: str) -> str:
    return (
        "https://openapivts.koreainvestment.com:29443"
        if mode == "paper"
        else "https://openapi.koreainvestment.com:9443"
    )


def _kis_headers(access_token: str, tr_id: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "appkey": settings.SYSTEM_KIS_APP_KEY,
        "appsecret": settings.SYSTEM_KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
        "Content-Type": "application/json; charset=utf-8",
    }


async def get_orderbook(code: str) -> dict:
    """10단 호가. system KIS 키 없으면 빈 결과."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return {"code": code, "asks": [], "bids": []}

    redis = await get_redis()
    cache_key = f"orderbook:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            f"{_base(settings.SYSTEM_KIS_MODE)}/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            headers=_kis_headers(token, "FHKST01010200"),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        resp.raise_for_status()
        out = resp.json().get("output1", {})

    asks = [
        {"price": int(out.get(f"askp{i}", 0)), "qty": int(out.get(f"askp_rsqn{i}", 0))}
        for i in range(1, 11)
    ]
    bids = [
        {"price": int(out.get(f"bidp{i}", 0)), "qty": int(out.get(f"bidp_rsqn{i}", 0))}
        for i in range(1, 11)
    ]
    data = {"code": code, "asks": asks, "bids": bids}
    ttl = 5 if _is_market_open() else 60
    await redis.setex(cache_key, ttl, json.dumps(data))
    return data


async def get_recent_trades(code: str) -> list[dict]:
    """최근 체결 20건. system KIS 키 없으면 빈 결과."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return []

    redis = await get_redis()
    cache_key = f"trades:{code}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(
            f"{_base(settings.SYSTEM_KIS_MODE)}/uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion",
            headers=_kis_headers(token, "FHKST01010300"),
            params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code},
        )
        resp.raise_for_status()
        output2 = resp.json().get("output2", [])

    trades = [
        {
            "time": row.get("stck_cntg_hour", ""),
            "price": int(row.get("stck_prpr", 0)),
            "volume": int(row.get("cntg_vol", 0)),
            "change": row.get("prdy_vrss_sign", "3"),
        }
        for row in output2[:20]
    ]
    ttl = 3 if _is_market_open() else 60
    await redis.setex(cache_key, ttl, json.dumps(trades))
    return trades


async def get_intraday_ohlcv(code: str, interval: str) -> list[dict]:
    """분봉 OHLCV. interval: '1min'|'5min'|'15min'|'1h'. system KIS 키 없으면 빈 결과."""
    if not settings.SYSTEM_KIS_APP_KEY:
        return []

    redis = await get_redis()
    cache_key = f"intraday:{code}:{interval}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    # KIS FID_HOUR_CLS_CODE: 0=1분, 1=3분, 5=5분, 10=10분, 15=15분, 30=30분, 60=60분
    hour_cls_map = {"1min": "0", "5min": "5", "15min": "15", "1h": "60"}
    hour_cls = hour_cls_map.get(interval, "0")

    token = await get_access_token(
        settings.SYSTEM_KIS_APP_KEY, settings.SYSTEM_KIS_APP_SECRET, settings.SYSTEM_KIS_MODE
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{_base(settings.SYSTEM_KIS_MODE)}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice",
            headers=_kis_headers(token, "FHKST03010200"),
            params={
                "FID_ETC_CLS_CODE": "",
                "FID_COND_MRKT_DIV_CODE": "J",
                "FID_INPUT_ISCD": code,
                "FID_INPUT_HOUR_1": "090000",
                "FID_PW_DATA_INCU_YN": "Y",
                "FID_HOUR_CLS_CODE": hour_cls,
            },
        )
        resp.raise_for_status()
        output2 = resp.json().get("output2", [])

    candles = [
        {
            "date": row.get("stck_bsop_date", ""),
            "time": row.get("stck_cntg_hour", ""),
            "open": int(row.get("stck_oprc", 0)),
            "high": int(row.get("stck_hgpr", 0)),
            "low": int(row.get("stck_lwpr", 0)),
            "close": int(row.get("stck_prpr", 0)),
            "volume": int(row.get("cntg_vol", 0)),
        }
        for row in output2
    ]
    await redis.setex(cache_key, 60, json.dumps(candles))
    return candles
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_phase2.py -v 2>&1 | tail -20
```
Expected: `6 passed`

- [ ] **Step 6: 커밋**

```bash
git add backend/services/kis_market_service.py backend/core/config.py tests/test_phase2.py
git commit -m "feat: KIS REST market service — orderbook, trades, intraday OHLCV"
```

---

## Task 3: 분봉 차트 지원 (stocks 라우트 확장)

**Files:**
- Modify: `backend/api/routes/stocks.py`
- Test: `tests/test_phase2.py` (Task 3 섹션 추가)

분봉 interval(`1min/5min/15min/1h`)은 `kis_market_service.get_intraday_ohlcv`를 사용하고,  
일봉 이상(`day/week/month`)은 기존 `market_service.get_ohlcv_cached`를 사용한다.  
period `1d`는 당일 데이터(일봉 조회 days=1).

- [ ] **Step 1: 테스트 추가**

```python
# ─── Task 3: Chart Interval Extension ────────────────────────────────────────

@pytest.mark.asyncio
async def test_chart_intraday_returns_empty_without_system_key(client):
    """분봉 요청 시 system KIS 키가 없으면 빈 data를 반환한다 (200)."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        response = await client.get("/stocks/005930/chart?period=1d&interval=1min")
    assert response.status_code == 200
    assert response.json()["data"] == []


@pytest.mark.asyncio
async def test_chart_1d_period_accepted(client):
    """period=1d가 유효한 값으로 받아들여진다."""
    with patch("services.market_service.get_ohlcv_from_pykrx", return_value=[]):
        with patch("services.market_service.get_redis") as mock_get_redis:
            mock_redis = AsyncMock()
            mock_redis.get.return_value = None
            mock_redis.setex = AsyncMock()
            mock_get_redis.return_value = mock_redis
            response = await client.get("/stocks/005930/chart?period=1d&interval=day")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chart_invalid_interval_rejected(client):
    """지원하지 않는 interval은 422를 반환한다."""
    response = await client.get("/stocks/005930/chart?interval=2min")
    assert response.status_code == 422
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_phase2.py::test_chart_intraday_returns_empty_without_system_key -v
```
Expected: FAIL — `422 Unprocessable Entity` (period `1d` 또는 interval `1min` 미지원)

- [ ] **Step 3: stocks.py 수정**

`backend/api/routes/stocks.py` 전체 교체:

```python
from fastapi import APIRouter, Depends, Query, Request

from api.middleware.rate_limit import limiter
from services import kis_market_service, market_service

router = APIRouter()

_INTRADAY_INTERVALS = {"1min", "5min", "15min", "1h"}


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
    period: str = Query("1m", pattern="^(1d|1w|1m|3m|1y)$"),
    interval: str = Query("day", pattern="^(1min|5min|15min|1h|day|week|month)$"),
):
    if interval in _INTRADAY_INTERVALS:
        data = await kis_market_service.get_intraday_ohlcv(code, interval)
    else:
        # period '1d' → 1 day window; reuse existing period_days by passing '1d'
        data = await market_service.get_ohlcv_cached(code, period, interval)
    return {"code": code, "period": period, "interval": interval, "data": data}


@router.get("/{code}/orderbook")
@limiter.limit("100/minute")
async def get_orderbook(request: Request, code: str):
    return await kis_market_service.get_orderbook(code)


@router.get("/{code}/trades")
@limiter.limit("100/minute")
async def get_recent_trades(request: Request, code: str):
    return await kis_market_service.get_recent_trades(code)


@router.get("/{code}")
@limiter.limit("100/minute")
async def get_stock_detail(request: Request, code: str):
    return await market_service.get_stock_current_price(code)
```

- [ ] **Step 4: market_service.py에 '1d' period 추가**

`backend/services/market_service.py` 내 `get_ohlcv_from_pykrx` 함수의 `period_days` 딕셔너리 수정:

```python
    period_days = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "1y": 365}
```

(기존 `"6m": 180, "3y": 1095` 제거 — TRD spec에 없음)

- [ ] **Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_phase2.py -v 2>&1 | tail -20
```
Expected: 9 passed

- [ ] **Step 6: 커밋**

```bash
git add backend/api/routes/stocks.py backend/services/market_service.py tests/test_phase2.py
git commit -m "feat: chart intraday intervals (1min/5min/15min/1h) + 1d period + orderbook/trades routes"
```

---

## Task 4: KIS WebSocket Pool

**Files:**
- Create: `backend/services/websocket_service.py`
- Modify: `backend/requirements.txt` (websockets 명시)
- Test: `tests/test_phase2.py` (Task 4 섹션 추가)

**설계:**
- `KISWebSocketPool` 싱글톤: 41종목씩 배치로 세션 관리
- `subscribe(code)` / `unsubscribe(code)`: 구독 카운트 관리
- `on_execution_message(raw)`: 파이프+캐럿 파싱 → Redis `stock:{code}` 채널 publish
- `start()` / `stop()`: lifespan에서 호출
- system KIS 키 없으면 graceful no-op

- [ ] **Step 1: requirements.txt에 websockets 추가**

```
websockets==12.0
sse-starlette==2.1.3
```

```bash
pip install websockets==12.0 sse-starlette==2.1.3
```

- [ ] **Step 2: 테스트 추가**

```python
# ─── Task 4: KIS WebSocket Pool ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pool_noop_when_no_system_key():
    """system KIS 키가 없으면 subscribe가 no-op이다."""
    with patch("services.websocket_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()
        # subscribe should not raise even without keys
        await pool.subscribe("005930")
        assert pool.subscription_count("005930") == 0


@pytest.mark.asyncio
async def test_pool_subscription_count():
    """subscribe/unsubscribe가 카운트를 올바르게 관리한다."""
    with patch("services.websocket_service.settings") as mock_settings, \
         patch("services.websocket_service.get_approval_key", return_value="appkey"):
        mock_settings.SYSTEM_KIS_APP_KEY = "key"
        mock_settings.SYSTEM_KIS_APP_SECRET = "secret"
        mock_settings.SYSTEM_KIS_MODE = "paper"
        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()
        # _get_session 반환 None → _send_subscribe가 실제 WS 연결 시도 안 함
        pool._get_session = AsyncMock(return_value=None)
        pool._subscriptions["005930"] = 0

        await pool.subscribe("005930")
        assert pool.subscription_count("005930") == 1

        await pool.subscribe("005930")
        assert pool.subscription_count("005930") == 2

        await pool.unsubscribe("005930")
        assert pool.subscription_count("005930") == 1


def test_parse_execution_message():
    """H0STCNT0 메시지를 파싱해 체결 딕셔너리로 변환한다."""
    from services.websocket_service import _parse_execution_msg
    # 0|H0STCNT0|001|code^time^price^sign^change^rate^...^volume
    raw = "0|H0STCNT0|001|005930^093000^60100^2^100^0.17^0^0^0^0^0^0^1000"
    result = _parse_execution_msg(raw)
    assert result is not None
    assert result["code"] == "005930"
    assert result["price"] == 60100
    assert result["volume"] == 1000
    assert result["time"] == "093000"
    assert result["type"] == "execution"


def test_parse_execution_message_pingpong():
    """PINGPONG 메시지는 None을 반환한다."""
    from services.websocket_service import _parse_execution_msg
    assert _parse_execution_msg("1|PINGPONG|...") is None


@pytest.mark.asyncio
async def test_pool_publishes_to_redis_on_message():
    """메시지 수신 시 Redis stock:{code} 채널에 발행한다."""
    mock_redis = AsyncMock()
    raw = "0|H0STCNT0|001|005930^093000^60100^2^100^0.17^0^0^0^0^0^0^1000"

    with patch("services.websocket_service.get_redis", return_value=mock_redis):
        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()
        await pool._on_raw_message(raw)

    mock_redis.publish.assert_called_once()
    channel, payload = mock_redis.publish.call_args[0]
    assert channel == "stock:005930"
    import json as _json
    data = _json.loads(payload)
    assert data["price"] == 60100
```

- [ ] **Step 3: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_phase2.py::test_parse_execution_message -v
```
Expected: `ImportError`

- [ ] **Step 4: websocket_service.py 구현**

```python
# backend/services/websocket_service.py
import asyncio
import json
import logging
from collections import defaultdict

import websockets

from core.config import settings
from core.redis_client import get_redis
from services.kis_token_service import get_approval_key

logger = logging.getLogger(__name__)

_KIS_WS_PAPER = "ws://ops.koreainvestment.com:31000/"
_KIS_WS_REAL = "ws://ops.koreainvestment.com:21000/"

MAX_PER_SESSION = 41


def _ws_url(mode: str) -> str:
    return _KIS_WS_PAPER if mode == "paper" else _KIS_WS_REAL


def _parse_execution_msg(raw: str) -> dict | None:
    """H0STCNT0 파이프-캐럿 포맷 파싱. PINGPONG 등 비데이터는 None."""
    if raw.startswith("1"):
        return None
    parts = raw.split("|")
    if len(parts) < 4 or parts[1] != "H0STCNT0":
        return None
    fields = parts[3].split("^")
    if len(fields) < 13:
        return None
    return {
        "type": "execution",
        "code": fields[0],
        "time": fields[1],
        "price": int(fields[2]) if fields[2].lstrip("-").isdigit() else 0,
        "sign": fields[3],           # 1=상한/2=상승/3=보합/4=하한/5=하락
        "change": int(fields[4]) if fields[4].lstrip("-").isdigit() else 0,
        "change_rate": float(fields[5]) if fields[5] else 0.0,
        "volume": int(fields[12]) if fields[12].lstrip("-").isdigit() else 0,
    }


class KISWebSocketPool:
    def __init__(self) -> None:
        self._subscriptions: dict[str, int] = defaultdict(int)
        self._sessions: list[websockets.WebSocketClientProtocol] = []
        self._symbol_session: dict[str, int] = {}
        self._running = False

    def subscription_count(self, code: str) -> int:
        return self._subscriptions.get(code, 0)

    async def subscribe(self, code: str) -> None:
        if not settings.SYSTEM_KIS_APP_KEY:
            return
        self._subscriptions[code] += 1
        if self._subscriptions[code] == 1:
            await self._send_subscribe(code, tr_type="1")

    async def unsubscribe(self, code: str) -> None:
        if not settings.SYSTEM_KIS_APP_KEY:
            return
        if self._subscriptions.get(code, 0) <= 0:
            return
        self._subscriptions[code] -= 1
        if self._subscriptions[code] == 0:
            await self._send_subscribe(code, tr_type="2")

    async def _send_subscribe(self, code: str, tr_type: str) -> None:
        approval_key = await get_approval_key(
            settings.SYSTEM_KIS_APP_KEY,
            settings.SYSTEM_KIS_APP_SECRET,
            settings.SYSTEM_KIS_MODE,
        )
        msg = {
            "header": {
                "approval_key": approval_key,
                "custtype": "P",
                "tr_type": tr_type,
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": "H0STCNT0", "tr_key": code}},
        }
        session = await self._get_session()
        if session:
            await session.send(json.dumps(msg))

    async def _get_session(self) -> "websockets.WebSocketClientProtocol | None":
        active = [s for s in self._sessions if not s.closed]
        subscribed_count = sum(1 for v in self._subscriptions.values() if v > 0)
        needed = (subscribed_count // MAX_PER_SESSION) + 1

        while len(active) < needed:
            try:
                ws = await websockets.connect(
                    _ws_url(settings.SYSTEM_KIS_MODE),
                    ping_interval=30,
                    ping_timeout=10,
                )
                active.append(ws)
                self._sessions = active
                asyncio.create_task(self._recv_loop(ws))
            except Exception as e:
                logger.warning("KIS WS 연결 실패: %s", e)
                return None

        self._sessions = active
        return active[-1] if active else None

    async def _recv_loop(self, ws: "websockets.WebSocketClientProtocol") -> None:
        try:
            async for raw in ws:
                await self._on_raw_message(str(raw))
        except Exception as e:
            logger.debug("KIS WS recv_loop 종료: %s", e)

    async def _on_raw_message(self, raw: str) -> None:
        data = _parse_execution_msg(raw)
        if data:
            redis = await get_redis()
            await redis.publish(f"stock:{data['code']}", json.dumps(data))

    async def stop(self) -> None:
        self._running = False
        for ws in self._sessions:
            try:
                await ws.close()
            except Exception:
                pass
        self._sessions.clear()


kis_pool = KISWebSocketPool()
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
python3 -m pytest tests/test_phase2.py -v 2>&1 | tail -25
```
Expected: 14 passed

- [ ] **Step 6: 커밋**

```bash
git add backend/services/websocket_service.py backend/requirements.txt tests/test_phase2.py
git commit -m "feat: KIS WebSocket Pool with Redis Pub/Sub publish + 41-symbol batching"
```

---

## Task 5: SSE 스트리밍 엔드포인트

**Files:**
- Create: `backend/api/routes/realtime.py`
- Modify: `backend/main.py`
- Test: `tests/test_phase2.py` (Task 5 섹션 추가)

SSE 엔드포인트 `GET /ws/stocks/{code}`:
1. `kis_pool.subscribe(code)` 호출
2. Redis Pub/Sub `stock:{code}` 채널 구독
3. 메시지를 `EventSourceResponse`로 클라이언트에 스트리밍
4. 클라이언트 disconnect 시 `kis_pool.unsubscribe(code)` + Redis 구독 해제

`sse-starlette`의 `EventSourceResponse`는 async generator를 받는다.

- [ ] **Step 1: 테스트 추가**

```python
# ─── Task 5: SSE Streaming Endpoint ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_sse_endpoint_streams_redis_messages(client):
    """Redis Pub/Sub 메시지가 SSE 스트림으로 나온다."""
    import asyncio

    # Redis pubsub mock: subscribe → receive one message → stop
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()

    received_messages = [
        {"type": "message", "channel": b"stock:005930", "data": b'{"type":"execution","code":"005930","price":60100}'},
        None,  # 스트림 종료 신호
    ]

    async def mock_listen():
        for msg in received_messages:
            if msg is None:
                return
            yield msg

    mock_pubsub.listen = mock_listen

    mock_redis = AsyncMock()
    mock_redis.pubsub.return_value = mock_pubsub

    with patch("api.routes.realtime.get_redis", return_value=mock_redis), \
         patch("api.routes.realtime.kis_pool") as mock_pool:
        mock_pool.subscribe = AsyncMock()
        mock_pool.unsubscribe = AsyncMock()

        async with client.stream("GET", "/ws/stocks/005930") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            lines = []
            async for line in response.aiter_lines():
                lines.append(line)
                if line.startswith("data:"):
                    break

    assert any("60100" in line for line in lines)
    mock_pool.subscribe.assert_called_once_with("005930")


@pytest.mark.asyncio
async def test_sse_endpoint_unsubscribes_on_disconnect(client):
    """/ws/stocks/{code} 클라이언트 종료 시 pool.unsubscribe가 호출된다."""
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()

    async def mock_listen_empty():
        return
        yield  # make it a generator

    mock_pubsub.listen = mock_listen_empty

    mock_redis = AsyncMock()
    mock_redis.pubsub.return_value = mock_pubsub

    with patch("api.routes.realtime.get_redis", return_value=mock_redis), \
         patch("api.routes.realtime.kis_pool") as mock_pool:
        mock_pool.subscribe = AsyncMock()
        mock_pool.unsubscribe = AsyncMock()

        async with client.stream("GET", "/ws/stocks/005930") as response:
            pass

    mock_pool.unsubscribe.assert_called_once_with("005930")
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
python3 -m pytest tests/test_phase2.py::test_sse_endpoint_streams_redis_messages -v
```
Expected: `ImportError` 또는 `404 Not Found`

- [ ] **Step 3: realtime.py 구현**

```python
# backend/api/routes/realtime.py
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from core.redis_client import get_redis
from services.websocket_service import kis_pool

router = APIRouter()


@router.get("/ws/stocks/{code}")
async def stock_stream(code: str) -> EventSourceResponse:
    async def event_generator() -> AsyncGenerator[dict, None]:
        redis = await get_redis()
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"stock:{code}")
        await kis_pool.subscribe(code)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield {"data": message["data"].decode() if isinstance(message["data"], bytes) else message["data"]}
        finally:
            await kis_pool.unsubscribe(code)
            await pubsub.unsubscribe(f"stock:{code}")

    return EventSourceResponse(event_generator())
```

- [ ] **Step 4: main.py에 realtime 라우터 등록 + pool lifespan 추가**

`backend/main.py` 전체:

```python
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.middleware.rate_limit import limiter
from api.routes import auth as auth_router
from api.routes import stocks as stocks_router
from api.routes import realtime as realtime_router
from core.config import settings
from core.redis_client import close_redis
from services.websocket_service import kis_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await kis_pool.stop()
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

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(stocks_router.router, prefix="/stocks", tags=["stocks"])
app.include_router(realtime_router.router, tags=["realtime"])


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: 전체 테스트 통과 확인**

```bash
python3 -m pytest -v 2>&1 | tail -30
```
Expected: `30 passed` (기존) + Task 5 신규 테스트 passed.  
신규 테스트 2개 포함 총 **32 passed** 이상.

- [ ] **Step 6: 커밋**

```bash
git add backend/api/routes/realtime.py backend/main.py tests/test_phase2.py
git commit -m "feat: SSE streaming endpoint /ws/stocks/{code} via Redis Pub/Sub"
```

---

## Task 6: 전체 연결 검증 + .env.example 업데이트

**Files:**
- Modify: `.env.example`
- Modify: `docs/progress.md`
- Test: 기존 전체 테스트 재실행

- [ ] **Step 1: .env.example에 Phase 2 설정 추가**

`.env.example` 하단에 추가:

```bash
# Phase 2 — 실시간 시세 (선택: 없으면 orderbook/trades/분봉 빈 결과)
SYSTEM_KIS_APP_KEY=
SYSTEM_KIS_APP_SECRET=
SYSTEM_KIS_MODE=paper
```

- [ ] **Step 2: 전체 테스트 실행**

```bash
python3 -m pytest -q 2>&1 | tail -10
```
Expected: 모든 테스트 passed, 0 failed

- [ ] **Step 3: docs/progress.md Phase 2 섹션 업데이트**

`docs/progress.md`에서 `## Phase 2` 섹션의 `🔲 미시작`을 `✅`로 변경하고 구현 항목 체크:

```markdown
## Phase 2 — 실시간 시세 + WebSocket ✅

**완료일:** 2026-06-03 | **테스트:** 32+ passed
```

각 항목에 파일 경로와 ✅ 추가:

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| KIS OAuth 토큰 캐시 (access_token + approval_key) | `backend/services/kis_token_service.py` | ✅ |
| KIS REST 호가·체결·분봉 | `backend/services/kis_market_service.py` | ✅ |
| KIS WebSocket Pool (41종목 배치, Redis Pub/Sub) | `backend/services/websocket_service.py` | ✅ |
| SSE 스트리밍 `/ws/stocks/{code}` | `backend/api/routes/realtime.py` | ✅ |
| `/stocks/{code}/orderbook` | `backend/api/routes/stocks.py` | ✅ |
| `/stocks/{code}/trades` | `backend/api/routes/stocks.py` | ✅ |
| 분봉 차트 (1min/5min/15min/1h) + period 1d | `backend/api/routes/stocks.py` | ✅ |

- [ ] **Step 4: 최종 커밋**

```bash
git add .env.example docs/progress.md
git commit -m "docs: update phase2 progress + env.example SYSTEM_KIS vars"
```

---

## 구현 후 확인 사항

**로컬에서 실시간 테스트 (KIS 키 보유 시):**
```bash
# .env에 SYSTEM_KIS_* 설정 후
docker-compose up -d postgres redis
cd backend && alembic upgrade head && uvicorn main:app --reload

# 다른 터미널
curl -N http://localhost:8000/ws/stocks/005930
# SSE 스트림 수신 확인

curl http://localhost:8000/stocks/005930/orderbook
curl http://localhost:8000/stocks/005930/trades
curl "http://localhost:8000/stocks/005930/chart?period=1d&interval=1min"
```

**KIS 키 없이 동작 확인:**
- `/ws/stocks/{code}` → 200, 빈 스트림 (메시지 없음)
- `/stocks/{code}/orderbook` → `{"code":"...","asks":[],"bids":[]}`
- `/stocks/{code}/trades` → `[]`
- `/stocks/{code}/chart?interval=1min` → `{"data":[]}`
