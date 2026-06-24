# backend/api/routes/watchlist.py
import re
import uuid as _uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
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
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(status_code=409, detail="이미 관심종목에 추가된 종목입니다.")
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
    if "target_price_high" in body.model_fields_set:
        item.target_price_high = body.target_price_high
    if "target_price_low" in body.model_fields_set:
        item.target_price_low = body.target_price_low
    if body.sort_order is not None:
        item.sort_order = body.sort_order
    if "group_id" in body.model_fields_set and body.group_id is not None:
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
