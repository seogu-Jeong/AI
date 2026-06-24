# tests/test_watchlist.py
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.deps import get_current_user, get_db


def _mock_user():
    u = MagicMock()
    u.id = uuid.uuid4()
    u.mode = "paper"
    return u


def _setup(app, user, db):
    """get_current_user + get_db 오버라이드 설정."""
    app.dependency_overrides[get_current_user] = lambda: user
    async def _db():
        yield db
    app.dependency_overrides[get_db] = _db


def _teardown(app):
    app.dependency_overrides.clear()


def _make_db(*execute_returns):
    """execute_returns: 각 execute() 호출에 대한 반환값 (list or single object or None)."""
    db = AsyncMock()
    db.add = MagicMock()

    side_effects = []
    for val in execute_returns:
        r = MagicMock()
        if isinstance(val, list):
            r.scalars.return_value.all.return_value = val
            r.scalar_one_or_none.return_value = val[0] if val else None
        elif val is None:
            r.scalars.return_value.all.return_value = []
            r.scalar_one_or_none.return_value = None
        else:
            r.scalar_one_or_none.return_value = val
            r.scalars.return_value.all.return_value = [val]
        side_effects.append(r)
    db.execute.side_effect = side_effects
    return db


def _mock_group(name="테스트 그룹"):
    g = MagicMock()
    g.id = uuid.uuid4()
    g.name = name
    g.sort_order = 0
    g.created_at = datetime.now(timezone.utc)
    return g


def _mock_item(stock_code="005930", group_id=None, iid=None):
    i = MagicMock()
    i.id = iid or uuid.uuid4()
    i.group_id = group_id or uuid.uuid4()
    i.user_id = uuid.uuid4()
    i.stock_code = stock_code
    i.stock_name = "삼성전자"
    i.target_price_high = None
    i.target_price_low = None
    i.sort_order = 0
    return i


async def test_get_groups_empty(client):
    """그룹 없는 유저 → 빈 리스트."""
    from main import app
    user = _mock_user()
    db = _make_db([], [])   # groups query → [], items query → []
    _setup(app, user, db)
    resp = await client.get("/watchlist/groups")
    _teardown(app)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_group(client):
    """그룹 생성 → 201 + name/id 반환."""
    from main import app
    user = _mock_user()
    db = _make_db()
    group_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    async def mock_refresh(obj):
        obj.id = group_id
        obj.created_at = created_at
    db.refresh = mock_refresh

    _setup(app, user, db)
    resp = await client.post("/watchlist/groups", json={"name": "주목 종목"})
    _teardown(app)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "주목 종목"
    assert "id" in data


async def test_add_item_to_group(client):
    """그룹에 종목 추가 → 201 + stock_code 반환."""
    from main import app
    user = _mock_user()
    group = _mock_group()
    # execute 1: group 존재 확인, execute 2: 중복 없음(None)
    db = _make_db(group, None)
    item_id = uuid.uuid4()

    async def mock_refresh(obj):
        obj.id = item_id
    db.refresh = mock_refresh

    _setup(app, user, db)
    resp = await client.post("/watchlist/items", json={
        "group_id": str(group.id),
        "stock_code": "005930",
        "stock_name": "삼성전자",
    })
    _teardown(app)
    assert resp.status_code == 201
    assert resp.json()["stock_code"] == "005930"


async def test_add_duplicate_item(client):
    """같은 종목 중복 추가 → 409."""
    from main import app
    user = _mock_user()
    group = _mock_group()
    existing_item = _mock_item()
    # execute 1: group 존재, execute 2: 중복 발견
    db = _make_db(group, existing_item)
    _setup(app, user, db)
    resp = await client.post("/watchlist/items", json={
        "group_id": str(group.id),
        "stock_code": "005930",
    })
    _teardown(app)
    assert resp.status_code == 409


async def test_update_item_target_price(client):
    """목표가 수정 → 200 {"updated": True}."""
    from main import app
    user = _mock_user()
    item = _mock_item()
    db = _make_db(item)
    _setup(app, user, db)
    resp = await client.put(
        f"/watchlist/items/{item.id}",
        json={"target_price_high": 80000.0},
    )
    _teardown(app)
    assert resp.status_code == 200
    assert resp.json()["updated"] is True


async def test_delete_group(client):
    """그룹 삭제 → 200 {"deleted": True}."""
    from main import app
    user = _mock_user()
    group = _mock_group()
    db = _make_db(group)
    _setup(app, user, db)
    resp = await client.delete(f"/watchlist/groups/{group.id}")
    _teardown(app)
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True
