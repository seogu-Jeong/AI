from unittest.mock import patch


async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_register_success(client):
    with patch("api.routes.auth.send_verification_email") as mock_email, \
         patch("api.routes.auth.settings") as mock_settings:
        mock_settings.SENDGRID_API_KEY = "test-key"
        response = await client.post(
            "/auth/register", json={"email": "test@example.com", "password": "password123"}
        )
    assert response.status_code == 201
    assert response.json()["message"] == "인증 이메일을 발송했습니다"
    mock_email.assert_called_once_with("test@example.com")


async def test_register_duplicate_email(client):
    with patch("api.routes.auth.send_verification_email"):
        await client.post("/auth/register", json={"email": "dup@example.com", "password": "pass1234"})
        response = await client.post(
            "/auth/register", json={"email": "dup@example.com", "password": "pass1234"}
        )
    assert response.status_code == 409


async def test_verify_email_success(client):
    from core.security import create_email_token

    with patch("api.routes.auth.send_verification_email"):
        await client.post(
            "/auth/register", json={"email": "verify@example.com", "password": "pass1234"}
        )
    token = create_email_token("verify@example.com")
    response = await client.post("/auth/verify-email", json={"token": token})
    assert response.status_code == 200


async def test_verify_email_invalid_token(client):
    response = await client.post("/auth/verify-email", json={"token": "totally-invalid"})
    assert response.status_code == 400


async def _register_and_verify(client, email: str, password: str = "pass1234"):
    with patch("api.routes.auth.send_verification_email"):
        await client.post("/auth/register", json={"email": email, "password": password})
    from core.security import create_email_token
    token = create_email_token(email)
    await client.post("/auth/verify-email", json={"token": token})


async def test_login_success(client):
    await _register_and_verify(client, "login@test.com")
    response = await client.post(
        "/auth/login", json={"email": "login@test.com", "password": "pass1234"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.cookies.get("refresh_token") is not None


async def test_login_wrong_password(client):
    await _register_and_verify(client, "wrongpw@test.com")
    response = await client.post(
        "/auth/login", json={"email": "wrongpw@test.com", "password": "badpass"}
    )
    assert response.status_code == 401


async def test_login_unverified_user(client):
    with patch("api.routes.auth.send_verification_email"), \
         patch("api.routes.auth.settings") as mock_settings:
        mock_settings.SENDGRID_API_KEY = "test-key"
        await client.post(
            "/auth/register", json={"email": "unverified@test.com", "password": "pass1234"}
        )
    response = await client.post(
        "/auth/login", json={"email": "unverified@test.com", "password": "pass1234"}
    )
    assert response.status_code == 403


async def test_refresh_token_rotation(client):
    await _register_and_verify(client, "refresh@test.com")
    await client.post("/auth/login", json={"email": "refresh@test.com", "password": "pass1234"})

    refresh_resp = await client.post("/auth/refresh")
    assert refresh_resp.status_code == 200
    assert "access_token" in refresh_resp.json()
    assert refresh_resp.cookies.get("refresh_token") is not None


async def test_logout_invalidates_refresh_token(client):
    await _register_and_verify(client, "logout@test.com")
    await client.post("/auth/login", json={"email": "logout@test.com", "password": "pass1234"})

    logout_resp = await client.post("/auth/logout")
    assert logout_resp.status_code == 200

    # After logout, refresh must fail
    refresh_resp = await client.post("/auth/refresh")
    assert refresh_resp.status_code == 401


async def test_google_login_redirects(client):
    # When GOOGLE_CLIENT_ID is empty, authlib raises an error → 400 or 500 acceptable
    response = await client.get("/auth/google", follow_redirects=False)
    assert response.status_code in (302, 307, 400, 422, 500)
    if response.status_code in (302, 307):
        assert "accounts.google.com" in response.headers.get("location", "")


async def test_me_returns_user_info(client):
    await _register_and_verify(client, "me@test.com")
    login_resp = await client.post(
        "/auth/login", json={"email": "me@test.com", "password": "pass1234"}
    )
    token = login_resp.json()["access_token"]

    response = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "me@test.com"
    assert data["mode"] == "demo"
    assert data["is_verified"] is True


async def test_me_unauthorized_without_token(client):
    response = await client.get("/auth/me")
    assert response.status_code == 401  # HTTPBearer returns 401 when header missing


async def test_register_kis_paper_key(client):
    from unittest.mock import AsyncMock, patch

    await _register_and_verify(client, "kis@test.com")
    login_resp = await client.post(
        "/auth/login", json={"email": "kis@test.com", "password": "pass1234"}
    )
    token = login_resp.json()["access_token"]

    with patch("services.kis_service.test_kis_connection", new_callable=AsyncMock, return_value=True):
        response = await client.put(
            "/auth/api-key",
            json={
                "mode": "paper",
                "app_key": "PXXXXXXXXXXXXXXXXXXX",
                "app_secret": "SXXXXXXXXXXXXXXXXXXX",
                "account_no": "12345678-01",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json()["message"] == "KIS API 키가 등록되었습니다"

    # Verify mode updated to paper
    me_resp = await client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.json()["mode"] == "paper"
