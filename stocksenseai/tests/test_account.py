import uuid
from unittest.mock import MagicMock, patch

from api.deps import get_current_user
from main import app


async def test_account_config_returns_system_mode_and_masked_account(client):
    user = MagicMock(id=uuid.uuid4())
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        with patch("api.routes.account.settings.SYSTEM_KIS_MODE", "real"), \
             patch("api.routes.account.settings.SYSTEM_KIS_ACCOUNT_NO", "12345678-01"):
            response = await client.get("/account/config")
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    assert response.json() == {
        "mode": "real",
        "account_no": "1234****-01",
    }


async def test_balance_ignores_query_mode_and_uses_system_mode(client):
    user = MagicMock(id=uuid.uuid4())
    app.dependency_overrides[get_current_user] = lambda: user
    try:
        with patch("api.routes.account.settings.SYSTEM_KIS_MODE", "paper"), \
             patch(
                 "api.routes.account.kis_account_service.get_account_balance",
                 return_value={"mode": "paper"},
             ) as get_balance:
            response = await client.get("/account/balance?mode=real")
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    get_balance.assert_awaited_once_with("paper")
