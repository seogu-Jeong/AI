# Phase 4-D — 관심종목 + 알림 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 관심종목 그룹/아이템 CRUD API + 목표가 알림 + 일일 손실 자동 차단 구현

**Architecture:** `routes/watchlist.py`가 WatchlistGroup/Item 모델에 직접 접근. `email_tasks.py`의 stub 2개(`check_price_alerts`, `check_daily_loss`)를 실제 로직으로 교체. Redis TTL 24시간 쿨다운으로 알림 중복 방지.

**Tech Stack:** FastAPI, SQLAlchemy 2 async, pykrx + Redis (market_service 재사용), Celery, SendGrid

---

## 파일 목록

| 파일 | 역할 |
|---|---|
| `backend/api/routes/watchlist.py` | 관심종목 CRUD 라우터 8개 (신규) |
| `backend/tasks/email_tasks.py` | check_price_alerts, check_daily_loss + send_price_alert_email 구현 (수정) |
| `tests/test_watchlist.py` | 관심종목 통합 테스트 6개 (신규) |
| `tests/test_alert_tasks.py` | 알림 태스크 단위 테스트 5개 (신규) |
| `backend/main.py` | watchlist 라우터 등록 (수정) |
| `docs/progress.md` | Phase 4-D 완료 표기 (수정) |

---

## Task 1: watchlist 라우터 (TDD)

**Files:**
- Create: `tests/test_watchlist.py`
- Create: `backend/api/routes/watchlist.py`
- Modify: `backend/main.py`

- [ ] **Step 1: `tests/test_watchlist.py` 작성**

```python
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
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_watchlist.py::test_get_groups_empty -v 2>&1 | tail -5
```

Expected: `404` 또는 `ImportError` (라우터 미등록)

- [ ] **Step 3: `backend/api/routes/watchlist.py` 생성**

```python
# backend/api/routes/watchlist.py
import re
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user, get_db
from api.middleware.rate_limit import limiter
from models.user import User
from models.watchlist import WatchlistGroup, WatchlistItem

router = APIRouter()


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    sort_order: int = 0


class GroupUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=50)
    sort_order: int | None = None


class ItemCreate(BaseModel):
    group_id: str
    stock_code: str
    stock_name: str | None = None
    target_price_high: float | None = Field(default=None, gt=0)
    target_price_low: float | None = Field(default=None, gt=0)
    sort_order: int = 0

    @field_validator("stock_code")
    @classmethod
    def valid_code(cls, v: str) -> str:
        if not re.fullmatch(r"\d{6}", v):
            raise ValueError("stock_code는 6자리 숫자여야 합니다.")
        return v


class ItemUpdate(BaseModel):
    target_price_high: float | None = Field(default=None, gt=0)
    target_price_low: float | None = Field(default=None, gt=0)
    sort_order: int | None = None
    group_id: str | None = None


def _serialize_group(g: WatchlistGroup, items: list) -> dict:
    return {
        "id": str(g.id),
        "name": g.name,
        "sort_order": g.sort_order,
        "created_at": g.created_at.isoformat() if g.created_at else None,
        "items": [_serialize_item(i) for i in items],
    }


def _serialize_item(i: WatchlistItem) -> dict:
    return {
        "id": str(i.id),
        "group_id": str(i.group_id),
        "stock_code": i.stock_code,
        "stock_name": i.stock_name,
        "target_price_high": float(i.target_price_high) if i.target_price_high else None,
        "target_price_low": float(i.target_price_low) if i.target_price_low else None,
        "sort_order": i.sort_order,
    }


@router.get("/groups")
@limiter.limit("60/minute")
async def get_groups(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(WatchlistGroup)
        .where(WatchlistGroup.user_id == user.id)
        .order_by(WatchlistGroup.sort_order, WatchlistGroup.created_at)
    )
    groups = res.scalars().all()

    res2 = await db.execute(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.sort_order, WatchlistItem.created_at)
    )
    items = res2.scalars().all()
    items_by_group: dict = {}
    for item in items:
        items_by_group.setdefault(str(item.group_id), []).append(item)

    return [_serialize_group(g, items_by_group.get(str(g.id), [])) for g in groups]


@router.post("/groups", status_code=201)
@limiter.limit("30/minute")
async def create_group(
    request: Request,
    body: GroupCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    group = WatchlistGroup(user_id=user.id, name=body.name, sort_order=body.sort_order)
    db.add(group)
    await db.commit()
    await db.refresh(group)
    return _serialize_group(group, [])


@router.put("/groups/{group_id}")
@limiter.limit("30/minute")
async def update_group(
    request: Request,
    group_id: str,
    body: GroupUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        gid = _uuid.UUID(group_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 group_id")
    res = await db.execute(
        select(WatchlistGroup).where(
            WatchlistGroup.id == gid, WatchlistGroup.user_id == user.id
        )
    )
    group = res.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail="그룹을 찾을 수 없습니다.")
    if body.name is not None:
        group.name = body.name
    if body.sort_order is not None:
        group.sort_order = body.sort_order
    await db.commit()
    return {"updated": True}


@router.delete("/groups/{group_id}")
@limiter.limit("30/minute")
async def delete_group(
    request: Request,
    group_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        gid = _uuid.UUID(group_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 group_id")
    res = await db.execute(
        select(WatchlistGroup).where(
            WatchlistGroup.id == gid, WatchlistGroup.user_id == user.id
        )
    )
    group = res.scalar_one_or_none()
    if not group:
        raise HTTPException(status_code=404, detail="그룹을 찾을 수 없습니다.")
    await db.delete(group)
    await db.commit()
    return {"deleted": True}


@router.get("/items")
@limiter.limit("60/minute")
async def get_items(
    request: Request,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    res = await db.execute(
        select(WatchlistItem)
        .where(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.sort_order, WatchlistItem.created_at)
    )
    return [_serialize_item(i) for i in res.scalars().all()]


@router.post("/items", status_code=201)
@limiter.limit("30/minute")
async def add_item(
    request: Request,
    body: ItemCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        gid = _uuid.UUID(body.group_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 group_id")
    res = await db.execute(
        select(WatchlistGroup).where(
            WatchlistGroup.id == gid, WatchlistGroup.user_id == user.id
        )
    )
    if not res.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="그룹을 찾을 수 없습니다.")
    dup = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.user_id == user.id,
            WatchlistItem.stock_code == body.stock_code,
        )
    )
    if dup.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 관심종목에 추가된 종목입니다.")
    item = WatchlistItem(
        group_id=gid,
        user_id=user.id,
        stock_code=body.stock_code,
        stock_name=body.stock_name,
        target_price_high=body.target_price_high,
        target_price_low=body.target_price_low,
        sort_order=body.sort_order,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return _serialize_item(item)


@router.put("/items/{item_id}")
@limiter.limit("30/minute")
async def update_item(
    request: Request,
    item_id: str,
    body: ItemUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        iid = _uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 item_id")
    res = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.id == iid, WatchlistItem.user_id == user.id
        )
    )
    item = res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="관심종목을 찾을 수 없습니다.")
    if body.target_price_high is not None:
        item.target_price_high = body.target_price_high
    if body.target_price_low is not None:
        item.target_price_low = body.target_price_low
    if body.sort_order is not None:
        item.sort_order = body.sort_order
    if body.group_id is not None:
        try:
            new_gid = _uuid.UUID(body.group_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="잘못된 group_id")
        g_res = await db.execute(
            select(WatchlistGroup).where(
                WatchlistGroup.id == new_gid, WatchlistGroup.user_id == user.id
            )
        )
        if not g_res.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="그룹을 찾을 수 없습니다.")
        item.group_id = new_gid
    await db.commit()
    return {"updated": True}


@router.delete("/items/{item_id}")
@limiter.limit("30/minute")
async def delete_item(
    request: Request,
    item_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        iid = _uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="잘못된 item_id")
    res = await db.execute(
        select(WatchlistItem).where(
            WatchlistItem.id == iid, WatchlistItem.user_id == user.id
        )
    )
    item = res.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="관심종목을 찾을 수 없습니다.")
    await db.delete(item)
    await db.commit()
    return {"deleted": True}
```

- [ ] **Step 4: `backend/main.py`에 watchlist 라우터 등록**

`backend/main.py`에서 `from api.routes import simulate as simulate_router` 아래에 추가:
```python
from api.routes import watchlist as watchlist_router
```

`app.include_router(simulate_router.router, ...)` 아래에 추가:
```python
app.include_router(watchlist_router.router, prefix="/watchlist", tags=["watchlist"])
```

- [ ] **Step 5: 테스트 실행 — 6개 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_watchlist.py -v 2>&1 | tail -15
```

Expected: `6 passed`

- [ ] **Step 6: 전체 회귀 테스트**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/ --tb=short -q 2>&1 | tail -5
```

Expected: `≥ 92 passed`, 기존 bcrypt 13개 실패 외 신규 실패 없음

- [ ] **Step 7: 커밋**

```bash
git add backend/api/routes/watchlist.py backend/main.py tests/test_watchlist.py
git commit -m "feat: watchlist router — groups/items CRUD (TDD)"
```

---

## Task 2: 알림 태스크 구현 (TDD)

**Files:**
- Create: `tests/test_alert_tasks.py`
- Modify: `backend/tasks/email_tasks.py`

- [ ] **Step 1: `tests/test_alert_tasks.py` 작성**

```python
# tests/test_alert_tasks.py
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_item(user_id, stock_code="005930", high=None, low=None):
    item = MagicMock()
    item.user_id = user_id
    item.stock_code = stock_code
    item.stock_name = "삼성전자"
    item.target_price_high = high
    item.target_price_low = low
    return item


def _make_alert_cfg(watchlist_price=True, daily_loss_limit=True):
    cfg = MagicMock()
    cfg.watchlist_price = watchlist_price
    cfg.daily_loss_limit = daily_loss_limit
    return cfg


def _mock_db_for_price_alerts(items, alert_cfgs):
    db = AsyncMock()
    items_result = MagicMock()
    items_result.scalars.return_value.all.return_value = items
    alerts_result = MagicMock()
    alerts_result.scalars.return_value.all.return_value = alert_cfgs
    db.execute.side_effect = [items_result, alerts_result]
    return db


async def test_price_alert_high_triggered():
    """현재가 ≥ 목표가 → send_price_alert_email.delay 호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg()
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_called_once()
    kwargs = mock_send.delay.call_args.kwargs
    assert kwargs["alert_type"] == "high"
    assert kwargs["stock_code"] == "005930"
    assert kwargs["current_price"] == 76000


async def test_price_alert_cooldown_skip():
    """Redis 쿨다운 키 있음 → delay 미호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg()
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])
    mock_redis = AsyncMock()
    mock_redis.exists.return_value = True  # 쿨다운 키 존재

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_not_called()


async def test_price_alert_setting_disabled():
    """watchlist_price=False → delay 미호출."""
    from tasks.email_tasks import _check_price_alerts_async

    user_id = uuid.uuid4()
    item = _make_item(user_id, high=75000.0)
    alert_cfg = _make_alert_cfg(watchlist_price=False)
    alert_cfg.user_id = user_id

    db = _mock_db_for_price_alerts([item], [alert_cfg])
    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_stock_current_price",
                new_callable=AsyncMock,
                return_value={"close": 76000},
            ):
                with patch("tasks.email_tasks.send_price_alert_email") as mock_send:
                    await _check_price_alerts_async()

    mock_send.delay.assert_not_called()


async def test_daily_loss_blocks_trading():
    """loss_pct > limit → trading_blocked=True + send_risk_alert.delay 호출."""
    from tasks.email_tasks import _check_daily_loss_async

    user_id = uuid.uuid4()

    # Mock trade result (user_id, mode)
    trade_result = MagicMock()
    trade_result.all.return_value = [(user_id, "paper")]

    # Mock alert settings (daily_loss_limit=True)
    alert_result = MagicMock()
    alert_cfg = _make_alert_cfg(daily_loss_limit=True)
    alert_result.scalar_one_or_none.return_value = alert_cfg

    db = AsyncMock()
    db.execute.side_effect = [trade_result, alert_result]

    mock_settings = MagicMock()
    mock_settings.trading_blocked = False
    mock_settings.daily_loss_limit_pct = 5.0

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_or_create_settings",
                new_callable=AsyncMock,
                return_value=mock_settings,
            ):
                with patch(
                    "tasks.email_tasks._get_today_loss",
                    new_callable=AsyncMock,
                    return_value=600_000,       # 60만원 손실
                ):
                    with patch(
                        "tasks.email_tasks._get_portfolio_total",
                        new_callable=AsyncMock,
                        return_value=10_000_000,  # 포트폴리오 1천만원
                    ):
                        with patch("tasks.email_tasks.send_risk_alert") as mock_alert:
                            await _check_daily_loss_async()

    assert mock_settings.trading_blocked is True
    mock_alert.delay.assert_called_once()


async def test_daily_loss_within_limit():
    """loss_pct ≤ limit → 차단 없음."""
    from tasks.email_tasks import _check_daily_loss_async

    user_id = uuid.uuid4()

    trade_result = MagicMock()
    trade_result.all.return_value = [(user_id, "paper")]

    alert_result = MagicMock()
    alert_cfg = _make_alert_cfg(daily_loss_limit=True)
    alert_result.scalar_one_or_none.return_value = alert_cfg

    db = AsyncMock()
    db.execute.side_effect = [trade_result, alert_result]

    mock_settings = MagicMock()
    mock_settings.trading_blocked = False
    mock_settings.daily_loss_limit_pct = 5.0

    mock_redis = AsyncMock()
    mock_redis.exists.return_value = False

    with patch("tasks.email_tasks.AsyncSessionLocal") as mock_sess:
        mock_sess.return_value.__aenter__ = AsyncMock(return_value=db)
        mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("tasks.email_tasks.get_redis", return_value=mock_redis):
            with patch(
                "tasks.email_tasks.get_or_create_settings",
                new_callable=AsyncMock,
                return_value=mock_settings,
            ):
                with patch(
                    "tasks.email_tasks._get_today_loss",
                    new_callable=AsyncMock,
                    return_value=100_000,        # 10만원 손실 (1%)
                ):
                    with patch(
                        "tasks.email_tasks._get_portfolio_total",
                        new_callable=AsyncMock,
                        return_value=10_000_000,
                    ):
                        with patch("tasks.email_tasks.send_risk_alert") as mock_alert:
                            await _check_daily_loss_async()

    assert mock_settings.trading_blocked is False
    mock_alert.delay.assert_not_called()
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_alert_tasks.py::test_price_alert_high_triggered -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name '_check_price_alerts_async'`

- [ ] **Step 3: `backend/tasks/email_tasks.py` 수정 — check_price_alerts 구현**

`email_tasks.py` 파일에서 기존 import 블록 끝 (`from tasks import celery_app` 아래)에 추가:

```python
import asyncio

from tasks import celery_app
```

그리고 파일 맨 아래 기존 stub들을 다음으로 교체:

```python
@celery_app.task
def check_price_alerts() -> None:
    """관심종목 목표가 도달 시 이메일 — APScheduler 5분 간격."""
    asyncio.run(_check_price_alerts_async())


async def _check_price_alerts_async() -> None:
    from core.database import AsyncSessionLocal
    from core.redis_client import get_redis
    from models.risk import AlertSettings
    from models.watchlist import WatchlistItem
    from services.market_service import get_stock_current_price
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        res = await db.execute(
            select(WatchlistItem).where(
                (WatchlistItem.target_price_high.isnot(None))
                | (WatchlistItem.target_price_low.isnot(None))
            )
        )
        items = res.scalars().all()
        if not items:
            return

        alert_res = await db.execute(
            select(AlertSettings).where(
                AlertSettings.user_id.in_([i.user_id for i in items])
            )
        )
        alert_map = {str(a.user_id): a for a in alert_res.scalars().all()}

        price_map: dict[str, int] = {}
        for code in {i.stock_code for i in items}:
            try:
                data = await get_stock_current_price(code)
                price_map[code] = data.get("close", 0)
            except Exception:
                pass

        redis = await get_redis()

        for item in items:
            uid_str = str(item.user_id)
            cfg = alert_map.get(uid_str)
            if cfg and not cfg.watchlist_price:
                continue

            current = price_map.get(item.stock_code, 0)
            if current == 0:
                continue

            if item.target_price_high and current >= float(item.target_price_high):
                key = f"price_alert:{uid_str}:{item.stock_code}:high"
                if not await redis.exists(key):
                    await redis.setex(key, 86400, "1")
                    send_price_alert_email.delay(
                        user_id=uid_str,
                        stock_code=item.stock_code,
                        stock_name=item.stock_name or item.stock_code,
                        current_price=current,
                        target_price=float(item.target_price_high),
                        alert_type="high",
                    )

            if item.target_price_low and current <= float(item.target_price_low):
                key = f"price_alert:{uid_str}:{item.stock_code}:low"
                if not await redis.exists(key):
                    await redis.setex(key, 86400, "1")
                    send_price_alert_email.delay(
                        user_id=uid_str,
                        stock_code=item.stock_code,
                        stock_name=item.stock_name or item.stock_code,
                        current_price=current,
                        target_price=float(item.target_price_low),
                        alert_type="low",
                    )


@celery_app.task
def check_daily_loss() -> None:
    """일일 손실 한도 체크 — APScheduler 10분 간격."""
    asyncio.run(_check_daily_loss_async())


async def _check_daily_loss_async() -> None:
    from datetime import date, datetime, timezone
    from core.database import AsyncSessionLocal
    from core.redis_client import get_redis
    from models.risk import AlertSettings, RiskSettings
    from models.trade import Trade
    from services.risk_service import (
        _get_portfolio_total,
        _get_today_loss,
        get_or_create_settings,
    )
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        today_start = datetime.combine(
            date.today(), datetime.min.time()
        ).replace(tzinfo=timezone.utc)

        res = await db.execute(
            select(Trade.user_id, Trade.mode)
            .where(
                Trade.order_type == "SELL",
                Trade.status == "FILLED",
                Trade.filled_at >= today_start,
            )
            .distinct()
        )
        user_mode_pairs = res.all()
        if not user_mode_pairs:
            return

        redis = await get_redis()

        for user_id, mode in user_mode_pairs:
            uid_str = str(user_id)

            settings = await get_or_create_settings(user_id, db)
            if settings.trading_blocked:
                continue

            redis_key = f"daily_loss_alert:{uid_str}"
            if await redis.exists(redis_key):
                continue

            alert_res = await db.execute(
                select(AlertSettings).where(AlertSettings.user_id == user_id)
            )
            alert_cfg = alert_res.scalar_one_or_none()
            if alert_cfg and not alert_cfg.daily_loss_limit:
                continue

            today_loss = await _get_today_loss(user_id, mode, db)
            portfolio_total = await _get_portfolio_total(user_id, mode, db)
            if portfolio_total == 0:
                continue

            loss_pct = today_loss / portfolio_total * 100
            limit_pct = float(settings.daily_loss_limit_pct)

            if loss_pct > limit_pct:
                settings.trading_blocked = True
                settings.blocked_at = datetime.now(timezone.utc)
                await db.commit()
                await redis.setex(redis_key, 86400, "1")
                reason = f"일일 손실 {loss_pct:.1f}% > 한도 {limit_pct:.1f}%"
                send_risk_alert.delay(uid_str, reason)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_price_alert_email(
    self,
    user_id: str,
    stock_code: str,
    stock_name: str,
    current_price: float,
    target_price: float,
    alert_type: str,
) -> None:
    """목표가/손절가 도달 이메일. alert_type: 'high' | 'low'"""
    to_email = _get_notification_email(user_id)
    if not to_email:
        return

    label = "목표가 도달" if alert_type == "high" else "손절가 도달"
    subject = f"[StockSenseAI] {stock_name}({stock_code}) {label}"
    body = (
        f"<p><strong>{stock_name}</strong>({stock_code}) {label}!</p>"
        f"<p>현재가: <strong>{int(current_price):,}원</strong> "
        f"/ 설정가: {int(target_price):,}원</p>"
    )

    try:
        from core.config import settings as app_settings
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        if not app_settings.SENDGRID_API_KEY:
            return
        msg = Mail(
            from_email=app_settings.FROM_EMAIL,
            to_emails=to_email,
            subject=subject,
            html_content=body,
        )
        client = SendGridAPIClient(app_settings.SENDGRID_API_KEY)
        asyncio.run(asyncio.to_thread(client.send, msg))
    except Exception as exc:
        raise self.retry(exc=exc)
```

Note: `email_tasks.py` 파일 상단 `import asyncio`가 이미 있는지 확인 후 없으면 추가.

- [ ] **Step 4: 테스트 실행 — 5개 통과 확인**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/test_alert_tasks.py -v 2>&1 | tail -12
```

Expected: `5 passed`

- [ ] **Step 5: 전체 회귀 테스트**

```bash
cd /Users/hwang/Gwang/Class/aiclass/FinalProject/backend && \
python3 -m pytest ../tests/ --tb=short -q 2>&1 | tail -5
```

Expected: `≥ 97 passed`, 기존 bcrypt 13개 외 신규 실패 없음

- [ ] **Step 6: 커밋**

```bash
git add backend/tasks/email_tasks.py tests/test_alert_tasks.py
git commit -m "feat: check_price_alerts + check_daily_loss 구현 — Redis 쿨다운, 자동 거래 차단 (TDD)"
```

---

## Task 3: progress.md 업데이트

**Files:**
- Modify: `docs/progress.md`

- [ ] **Step 1: Phase 4-D 완료 표기**

`docs/progress.md`에서:
```
### Phase 4-D — 관심종목 + 알림 🔲
```
→ 교체:
```
### Phase 4-D — 관심종목 + 알림 ✅

**완료일:** 2026-06-04

| 컴포넌트 | 파일 | 상태 |
|---|---|---|
| 관심종목 CRUD API | `backend/api/routes/watchlist.py` | ✅ |
| 목표가 알림 | `backend/tasks/email_tasks.py` (check_price_alerts) | ✅ |
| 일일 손실 자동 차단 | `backend/tasks/email_tasks.py` (check_daily_loss) | ✅ |
| 테스트 | `tests/test_watchlist.py` (6) + `tests/test_alert_tasks.py` (5) | ✅ |
```

- [ ] **Step 2: 커밋**

```bash
git add docs/progress.md
git commit -m "docs: Phase 4-D complete — watchlist CRUD + price/loss alerts"
```
