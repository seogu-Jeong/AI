import logging
import os
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import User, FREE_VIEW_LIMIT
from schemas import UserResponse, ViewResponse

logger = logging.getLogger("week5.users")
router = APIRouter()

INTERNAL_SECRET = os.getenv("INTERNAL_SECRET", "")


def verify_internal(x_internal_secret: str = Header(...)):
    if x_internal_secret != INTERNAL_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/users/{user_id}", response_model=UserResponse, dependencies=[Depends(verify_internal)])
async def get_user(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.post("/users/{user_id}/view", response_model=ViewResponse, dependencies=[Depends(verify_internal)])
async def record_view(user_id: str, db: AsyncSession = Depends(get_db)):
    """조회 기록 + 허용 여부 반환. Pro 유저는 무제한."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if user.is_pro:
        return ViewResponse(
            allowed=True,
            view_count=user.view_count,
            views_remaining=FREE_VIEW_LIMIT,
            is_pro=True,
        )

    if user.view_count < FREE_VIEW_LIMIT:
        user.view_count += 1
        await db.commit()
        logger.info("User %s view %d/%d", user_id, user.view_count, FREE_VIEW_LIMIT)

    allowed = user.view_count <= FREE_VIEW_LIMIT
    return ViewResponse(
        allowed=allowed,
        view_count=user.view_count,
        views_remaining=max(0, FREE_VIEW_LIMIT - user.view_count),
        is_pro=False,
    )


@router.post("/users/{user_id}/upgrade", dependencies=[Depends(verify_internal)])
async def upgrade_user(user_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_pro = True
    await db.commit()
    logger.info("User %s upgraded to Pro ✓", user_id)
    return {"success": True, "is_pro": True}
