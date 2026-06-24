from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from services.ai_service import get_prediction


async def test_get_prediction_uses_db_without_market_lookup():
    recorded_at = datetime.now(timezone.utc)
    row = SimpleNamespace(
        recorded_at=recorded_at,
        confidence=82,
        predicted_prices={
            "current_price": 73000,
            "bullish": [74000, 75000, 76000, 77000, 78000],
            "base": [73500, 74000, 74500, 75000, 75500],
            "bearish": [72000, 71500, 71000, 70500, 70000],
        },
    )
    result = MagicMock()
    result.scalar_one_or_none.return_value = row
    db = AsyncMock()
    db.execute.return_value = result
    redis = AsyncMock()
    redis.get.return_value = None

    with patch("services.ai_service.get_redis", return_value=redis), \
         patch("services.ai_service.get_ohlcv_cached", new_callable=AsyncMock) as market:
        response = await get_prediction("005930", db)

    market.assert_not_awaited()
    assert response["current_price"] == 73000
    assert response["confidence"] == 82.0
    assert response["source"] == "stored"
    assert response["predicted_at"] == recorded_at.isoformat()
