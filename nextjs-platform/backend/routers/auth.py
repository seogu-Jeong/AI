from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from models import User
from schemas import GoogleAuthRequest, UserResponse

router = APIRouter()


@router.post("/auth/google", response_model=UserResponse)
async def google_auth(payload: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.google_id == payload.google_id))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            google_id=payload.google_id,
            email=payload.email,
            name=payload.name,
            image=payload.image,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    else:
        # Update name/image in case they changed
        user.name = payload.name
        user.image = payload.image
        await db.commit()
        await db.refresh(user)

    return user
