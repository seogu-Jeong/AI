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


@pytest.mark.asyncio
async def test_get_access_token_raises_502_on_kis_error():
    """KIS가 에러를 반환하면 HTTPException 502를 발생시킨다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("services.kis_token_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
        mock_client_cls.return_value = mock_client

        from fastapi import HTTPException
        from services.kis_token_service import get_access_token
        with pytest.raises(HTTPException) as exc_info:
            await get_access_token("key", "secret", "paper")
    assert exc_info.value.status_code == 502


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
    fake_resp.json.return_value = {"rt_cd": "0", "output1": fake_output1}

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


def test_intraday_query_time_is_clamped_to_regular_session():
    """당일분봉 조회 기준 시간은 정규장 범위로 보정한다."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from services.kis_market_service import _intraday_query_time

    kst = ZoneInfo("Asia/Seoul")

    assert _intraday_query_time(datetime(2026, 6, 18, 8, 30, tzinfo=kst)) == "090000"
    assert _intraday_query_time(datetime(2026, 6, 18, 10, 0, tzinfo=kst)) == "100000"
    assert _intraday_query_time(datetime(2026, 6, 18, 23, 0, tzinfo=kst)) == "153000"


@pytest.mark.asyncio
async def test_get_recent_trades_returns_empty_when_no_system_key():
    """SYSTEM_KIS_APP_KEY가 없으면 빈 체결 목록을 반환한다."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        from services.kis_market_service import get_recent_trades
        result = await get_recent_trades("005930")
    assert result == []


@pytest.mark.asyncio
async def test_get_orderbook_raises_502_on_kis_error():
    """KIS 호가 API 실패 시 HTTPException 502를 반환한다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("services.kis_market_service.settings") as mock_settings, \
         patch("services.kis_market_service.get_access_token", return_value="tok"), \
         patch("services.kis_market_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_cls:
        mock_settings.SYSTEM_KIS_APP_KEY = "key"
        mock_settings.SYSTEM_KIS_APP_SECRET = "secret"
        mock_settings.SYSTEM_KIS_MODE = "paper"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))
        mock_cls.return_value = mock_client

        from fastapi import HTTPException
        from services.kis_market_service import get_orderbook
        with pytest.raises(HTTPException) as exc_info:
            await get_orderbook("005930")
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_get_intraday_ohlcv_rejects_invalid_interval():
    """알 수 없는 interval은 HTTPException 400을 반환한다."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = "key"
        mock_settings.SYSTEM_KIS_APP_SECRET = "secret"
        mock_settings.SYSTEM_KIS_MODE = "paper"
        from fastapi import HTTPException
        from services.kis_market_service import get_intraday_ohlcv
        with pytest.raises(HTTPException) as exc_info:
            await get_intraday_ohlcv("005930", "30min")
    assert exc_info.value.status_code == 400


# ─── Task 3: Chart + Orderbook/Trades Routes ─────────────────────────────────

@pytest.mark.asyncio
async def test_chart_intraday_returns_fallback_while_loading(client):
    """KIS 키가 없으면 일봉 fallback만 반환하고 재시도를 요구하지 않는다."""
    fallback = [{"date": "20260609", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10}]
    with patch("api.routes.stocks.settings") as mock_settings, \
         patch("api.routes.stocks.market_service.get_ohlcv_cached", return_value=fallback):
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        response = await client.get("/stocks/005930/chart?period=1d&interval=1min")
    assert response.status_code == 200
    assert response.json()["status"] == "fallback_only"
    assert response.json()["actual_interval"] == "day"
    assert response.json()["data"] == fallback


@pytest.mark.asyncio
async def test_chart_1d_period_accepted(client):
    """period=1d가 유효한 값으로 받아들여진다."""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.setex = AsyncMock()
    with patch("services.market_service.get_ohlcv_from_pykrx", return_value=[]), \
         patch("services.market_service.get_redis", return_value=mock_redis):
        response = await client.get("/stocks/005930/chart?period=1d&interval=day")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_chart_invalid_interval_rejected(client):
    """지원하지 않는 interval은 422를 반환한다."""
    response = await client.get("/stocks/005930/chart?interval=2min")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_orderbook_endpoint_returns_empty_without_system_key(client):
    """GET /stocks/{code}/orderbook: system 키 없으면 빈 asks/bids 반환."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        response = await client.get("/stocks/005930/orderbook")
    assert response.status_code == 200
    data = response.json()
    assert data["asks"] == []
    assert data["bids"] == []


@pytest.mark.asyncio
async def test_trades_endpoint_returns_empty_without_system_key(client):
    """GET /stocks/{code}/trades: system 키 없으면 빈 목록 반환."""
    with patch("services.kis_market_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        response = await client.get("/stocks/005930/trades")
    assert response.status_code == 200
    assert response.json() == []


# ─── Task 4: KIS WebSocket Pool ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pool_noop_when_no_system_key():
    """system KIS 키가 없으면 subscribe가 no-op이다."""
    with patch("services.websocket_service.settings") as mock_settings:
        mock_settings.SYSTEM_KIS_APP_KEY = ""
        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()
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
        pool._get_session = AsyncMock(return_value=None)  # prevent real WS connection
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
    import json as _json
    mock_redis = AsyncMock()
    raw = "0|H0STCNT0|001|005930^093000^60100^2^100^0.17^0^0^0^0^0^0^1000"

    with patch("services.websocket_service.get_redis", return_value=mock_redis):
        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()
        await pool._on_raw_message(raw)

    mock_redis.publish.assert_called_once()
    channel, payload = mock_redis.publish.call_args[0]
    assert channel == "stock:005930"
    data = _json.loads(payload)
    assert data["price"] == 60100


# ─── Task 5: SSE Streaming Endpoint ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_sse_endpoint_streams_redis_messages(client):
    """Redis Pub/Sub 메시지가 SSE 스트림으로 나온다."""
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.aclose = AsyncMock()

    received_messages = [
        {"type": "message", "channel": b"stock:005930", "data": b'{"type":"execution","code":"005930","price":60100}'},
    ]

    async def mock_listen():
        for msg in received_messages:
            yield msg

    mock_pubsub.listen = mock_listen

    mock_redis = AsyncMock()
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

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
                if any("60100" in l for l in lines):
                    break

    assert any("60100" in line for line in lines)
    mock_pool.subscribe.assert_called_once_with("005930")


@pytest.mark.asyncio
async def test_sse_endpoint_unsubscribes_on_disconnect(client):
    """클라이언트 종료 시 pool.unsubscribe와 pubsub.aclose가 호출된다."""
    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.aclose = AsyncMock()

    async def mock_listen_empty():
        return
        yield  # make it an async generator

    mock_pubsub.listen = mock_listen_empty

    mock_redis = AsyncMock()
    mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

    with patch("api.routes.realtime.get_redis", return_value=mock_redis), \
         patch("api.routes.realtime.kis_pool") as mock_pool:
        mock_pool.subscribe = AsyncMock()
        mock_pool.unsubscribe = AsyncMock()

        async with client.stream("GET", "/ws/stocks/005930") as response:
            async for _ in response.aiter_lines():
                break

    mock_pool.unsubscribe.assert_called_once_with("005930")
    mock_pubsub.aclose.assert_called_once()


# ─── Additional: Batch Calc & Session Routing ─────────────────────────────────

def test_pool_batch_calc_boundary():
    """41종목까지 세션 1개, 42번째부터 세션 2개 필요."""
    import math
    from services.websocket_service import MAX_PER_SESSION
    for n in range(1, MAX_PER_SESSION + 1):
        assert max(1, math.ceil(n / MAX_PER_SESSION)) == 1
    assert max(1, math.ceil((MAX_PER_SESSION + 1) / MAX_PER_SESSION)) == 2
    assert max(1, math.ceil((MAX_PER_SESSION * 2) / MAX_PER_SESSION)) == 2
    assert max(1, math.ceil((MAX_PER_SESSION * 2 + 1) / MAX_PER_SESSION)) == 3


@pytest.mark.asyncio
async def test_pool_unsubscribe_targets_original_session():
    """unsubscribe 메시지가 subscribe 시 사용한 세션으로 전송된다."""
    with patch("services.websocket_service.settings") as mock_settings, \
         patch("services.websocket_service.get_approval_key", return_value="appkey"):
        mock_settings.SYSTEM_KIS_APP_KEY = "key"
        mock_settings.SYSTEM_KIS_APP_SECRET = "secret"
        mock_settings.SYSTEM_KIS_MODE = "paper"

        from services.websocket_service import KISWebSocketPool
        pool = KISWebSocketPool()

        session_0 = AsyncMock()
        session_0.closed = False
        session_1 = AsyncMock()
        session_1.closed = False
        pool._sessions = [session_0, session_1]
        pool._get_session = AsyncMock(return_value=session_0)

        await pool.subscribe("005930")
        assert pool._symbol_session.get("005930") == 0

        pool._get_session = AsyncMock(return_value=session_1)
        await pool.unsubscribe("005930")

        assert session_0.send.call_count == 2  # subscribe + unsubscribe both on session_0
        session_1.send.assert_not_called()


@pytest.mark.asyncio
async def test_get_orderbook_raises_502_on_kis_api_level_error():
    """KIS rt_cd != '0' 응답은 HTTPException 502를 반환한다."""
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {"rt_cd": "1", "msg_cd": "MCA00004", "msg1": "잘못된 요청"}

    mock_redis = AsyncMock()
    mock_redis.get.return_value = None

    with patch("services.kis_market_service.settings") as mock_settings, \
         patch("services.kis_market_service.get_access_token", return_value="tok"), \
         patch("services.kis_market_service.get_redis", return_value=mock_redis), \
         patch("httpx.AsyncClient") as mock_cls:
        mock_settings.SYSTEM_KIS_APP_KEY = "key"
        mock_settings.SYSTEM_KIS_APP_SECRET = "secret"
        mock_settings.SYSTEM_KIS_MODE = "paper"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=fake_resp)
        mock_cls.return_value = mock_client

        from fastapi import HTTPException
        from services.kis_market_service import get_orderbook
        with pytest.raises(HTTPException) as exc_info:
            await get_orderbook("005930")
    assert exc_info.value.status_code == 502
