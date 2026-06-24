from datetime import datetime
from pydantic import BaseModel


class GoogleAuthRequest(BaseModel):
    google_id: str
    email: str
    name: str | None = None
    image: str | None = None


class UserResponse(BaseModel):
    id: str
    google_id: str
    email: str
    name: str | None
    image: str | None
    is_pro: bool
    view_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ViewResponse(BaseModel):
    allowed: bool
    view_count: int
    views_remaining: int
    is_pro: bool
