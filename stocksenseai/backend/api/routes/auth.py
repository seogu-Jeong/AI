import re
from datetime import datetime, timedelta, timezone
from typing import Literal

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_current_user
from api.middleware.rate_limit import limiter
from core.config import settings
from core.database import get_db
from core.security import (
    create_access_token,
    create_refresh_token,
    decode_email_token,
    encrypt_aes,
    hash_password,
    verify_password,
    verify_refresh_token,
)
from models.user import RefreshToken, User
from services.email_service import send_verification_email

router = APIRouter()

_REFRESH_COOKIE = "refresh_token"

oauth = OAuth()
oauth.register(
    name="google",
    client_id=settings.GOOGLE_CLIENT_ID,
    client_secret=settings.GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_policy(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("비밀번호는 최소 8자 이상이어야 합니다")
        if v.strip() != v:
            raise ValueError("비밀번호 앞뒤에 공백을 사용할 수 없습니다")
        return v


class VerifyEmailRequest(BaseModel):
    token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class KISKeyRequest(BaseModel):
    mode: Literal["paper", "real"]
    app_key: str
    app_secret: str
    account_no: str  # 형식: 12345678-01 (종합계좌번호 8자리-상품코드 2자리)

    @field_validator("account_no")
    @classmethod
    def validate_account_no(cls, v: str) -> str:
        if not re.match(r"^\d{8}-\d{2}$", v):
            raise ValueError("계좌번호 형식이 올바르지 않습니다 (예: 12345678-01)")
        return v


class MeResponse(BaseModel):
    id: str
    email: str
    is_verified: bool
    mode: str
    dark_mode: bool


@router.post("/register", status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="이미 등록된 이메일입니다")

    # SENDGRID 미설정(로컬 개발) 시 이메일 인증 생략하고 바로 활성화
    auto_verify = not settings.SENDGRID_API_KEY
    user = User(email=body.email, password_hash=hash_password(body.password), is_verified=auto_verify)
    db.add(user)
    await db.commit()

    if auto_verify:
        return {"message": "회원가입이 완료됐습니다. 바로 로그인하세요."}
    try:
        await send_verification_email(body.email)
    except Exception:
        # 이메일 발송 실패 — 계정은 생성됐으나 미인증 상태
        # 로그인 페이지에서 재발송을 안내
        return {
            "message": "회원가입은 완료됐으나 인증 이메일 발송에 실패했습니다. 로그인 페이지에서 재발송을 요청하세요.",
            "email_failed": True,
        }
    return {"message": "인증 이메일을 발송했습니다"}


@router.post("/verify-email")
async def verify_email(body: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    email = decode_email_token(body.token)
    if not email:
        raise HTTPException(status_code=400, detail="유효하지 않거나 만료된 토큰입니다")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다")

    user.is_verified = True
    await db.commit()
    return {"message": "이메일 인증이 완료되었습니다"}


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(
    request: Request, body: LoginRequest, response: Response, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not user.password_hash or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다")
    if not user.is_verified:
        raise HTTPException(status_code=403, detail="이메일 인증이 필요합니다")

    raw_rt, hashed_rt = create_refresh_token()
    expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=user.id, token_hash=hashed_rt, selector=raw_rt[:16], expires_at=expires))
    await db.commit()

    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=raw_rt,
        httponly=True,
        secure=settings.APP_ENV != "development",
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    return TokenResponse(access_token=create_access_token(str(user.id)))


@router.post("/refresh")
async def refresh_token(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    raw_rt = request.cookies.get(_REFRESH_COOKIE)
    if not raw_rt:
        raise HTTPException(status_code=401, detail="Refresh token missing")

    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.selector == raw_rt[:16],
            RefreshToken.revoked == False,  # noqa: E712
            RefreshToken.expires_at > datetime.now(timezone.utc),
        )
    )
    candidate = result.scalar_one_or_none()
    if not candidate or not verify_refresh_token(raw_rt, candidate.token_hash):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    matched = candidate

    matched.revoked = True
    raw_new, hashed_new = create_refresh_token()
    expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=matched.user_id, token_hash=hashed_new, selector=raw_new[:16], expires_at=expires))
    await db.commit()

    response.set_cookie(
        key=_REFRESH_COOKIE,
        value=raw_new,
        httponly=True,
        secure=settings.APP_ENV != "development",
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    return {"access_token": create_access_token(str(matched.user_id)), "token_type": "bearer"}


@router.post("/logout")
async def logout(request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    raw_rt = request.cookies.get(_REFRESH_COOKIE)
    if raw_rt:
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.selector == raw_rt[:16],
                RefreshToken.revoked == False,  # noqa: E712
            )
        )
        candidate = result.scalar_one_or_none()
        if candidate and verify_refresh_token(raw_rt, candidate.token_hash):
            candidate.revoked = True
            await db.commit()
    response.delete_cookie(_REFRESH_COOKIE)
    return {"message": "로그아웃되었습니다"}


@router.get("/google")
async def google_login(request: Request):
    return await oauth.google.authorize_redirect(request, settings.GOOGLE_REDIRECT_URI)


@router.get("/google/callback")
async def google_callback(
    request: Request, response: Response, db: AsyncSession = Depends(get_db)
):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise HTTPException(status_code=400, detail="Google OAuth 인증 실패")

    userinfo = token.get("userinfo", {})
    email = userinfo.get("email")
    google_id = userinfo.get("sub")
    if not email:
        raise HTTPException(status_code=400, detail="Google에서 이메일을 가져올 수 없습니다")

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user:
        if google_id:
            user.google_id = google_id
        user.is_verified = True
        user.password_hash = None  # 로컬 비밀번호 무효화 — pre-auth 계정 탈취 방지
    else:
        user = User(email=email, google_id=google_id, is_verified=True)
        db.add(user)
    await db.commit()
    await db.refresh(user)

    raw_rt, hashed_rt = create_refresh_token()
    expires = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    db.add(RefreshToken(user_id=user.id, token_hash=hashed_rt, selector=raw_rt[:16], expires_at=expires))
    await db.commit()

    access_token = create_access_token(str(user.id))
    redirect = RedirectResponse(url=f"{settings.FRONTEND_URL}/oauth-callback?token={access_token}")
    redirect.set_cookie(
        key=_REFRESH_COOKIE,
        value=raw_rt,
        httponly=True,
        secure=settings.APP_ENV != "development",
        samesite="lax",
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
    )
    return redirect


@router.get("/me", response_model=MeResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    return MeResponse(
        id=str(current_user.id),
        email=current_user.email,
        is_verified=current_user.is_verified,
        mode=current_user.mode,
        dark_mode=current_user.dark_mode,
    )


@router.put("/api-key")
async def register_api_key(
    body: KISKeyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    key_enc = encrypt_aes(body.app_key)
    secret_enc = encrypt_aes(body.app_secret)

    if body.mode == "paper":
        current_user.kis_paper_key_enc = key_enc
        current_user.kis_paper_secret_enc = secret_enc
        current_user.kis_paper_account_no = body.account_no
    else:
        current_user.kis_real_key_enc = key_enc
        current_user.kis_real_secret_enc = secret_enc
        current_user.kis_real_account_no = body.account_no

    test_result = None
    if settings.APP_ENV != "development":
        from services.kis_service import test_kis_connection
        ok = await test_kis_connection(body.app_key, body.app_secret, body.mode)
        test_result = ok
        if not ok:
            raise HTTPException(status_code=400, detail="KIS API 키 검증에 실패했습니다")

    current_user.mode = body.mode
    await db.commit()
    return {"message": "KIS API 키가 등록되었습니다", "test_result": test_result}
