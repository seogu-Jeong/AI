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
