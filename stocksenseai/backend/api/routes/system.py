# backend/api/routes/system.py
"""시스템 상태 진단 API — 인증 선택적, 민감 정보 비노출."""
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_optional_user
from api.middleware.rate_limit import limiter
from core.config import settings
from core.database import get_db
from models.ai_signal import AISignalHistory
from models.user import User
from services.kis_account_service import get_account_balance, mask_account_no

_KST = ZoneInfo("Asia/Seoul")

router = APIRouter()


async def _count_uploaded_predictions(db: AsyncSession) -> int:
    """DB에 저장된 LSTM 예측 건수를 반환. 실패 시 0 반환."""
    try:
        result = await db.execute(
            select(func.count()).select_from(AISignalHistory).where(AISignalHistory.predicted_prices.isnot(None))
        )
        return result.scalar() or 0
    except Exception:
        return 0


def _kis_configured() -> bool:
    return bool(settings.SYSTEM_KIS_APP_KEY and settings.SYSTEM_KIS_APP_SECRET and settings.SYSTEM_KIS_ACCOUNT_NO)


def _ki_mode_label(mode: str) -> str:
    return "모의투자" if mode == "paper" else "실계좌"


@router.get("/status")
@limiter.limit("30/minute")
async def get_system_status(
    request: Request,
    user: User | None = Depends(get_optional_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """시스템 상태 진단 — .env 원문 / KIS 키 미노출."""
    now = datetime.now(_KST)

    # 1. 백엔드 상태
    backend = {"ok": True, "message": "백엔드 정상"}

    # 2. 인증 상태
    if user:
        auth = {"logged_in": True, "email": user.email}
    else:
        auth = {"logged_in": False, "email": None}

    # 3. KIS 설정 상태
    configured = _kis_configured()
    if configured:
        mode = settings.SYSTEM_KIS_MODE
        masked_account = mask_account_no(settings.SYSTEM_KIS_ACCOUNT_NO) if user else None
        kis = {
            "mode": mode,
            "configured": True,
            "account_no": masked_account,
            "message": f"KIS {_ki_mode_label(mode)} 설정됨",
        }
    else:
        kis = {
            "mode": None,
            "configured": False,
            "account_no": None,
            "message": "KIS API 키 미설정 (.env 설정 필요)",
        }

    # 4. 계좌 잔고 상태
    if not user:
        account = {
            "ok": None,
            "holdings_count": None,
            "data_source": None,
            "message": "login_required",
        }
    elif not configured:
        account = {
            "ok": False,
            "holdings_count": None,
            "data_source": None,
            "message": "KIS 설정 미완료 — 잔고 조회 불가",
        }
    else:
        try:
            balance = await get_account_balance(settings.SYSTEM_KIS_MODE)
            holdings_count = len(balance.get("holdings", []))
            account = {
                "ok": True,
                "holdings_count": holdings_count,
                "data_source": balance.get("data_source", ""),
                "message": f"잔고 조회 성공 · 보유 {holdings_count}종목" if holdings_count > 0 else "잔고 조회 성공 · 보유 종목 없음",
            }
        except Exception as exc:
            detail = str(exc)
            # exc.detail (HTTPException) 추출
            if hasattr(exc, "detail"):
                detail = exc.detail
            # 민감 정보 노출 방지: app_key, app_secret 키워드 포함 시 제네릭 메시지
            if any(k in detail.lower() for k in ("key", "secret", "token", "bearer")):
                detail = "KIS 인증 오류 (설정 확인 필요)"
            account = {
                "ok": False,
                "holdings_count": None,
                "data_source": None,
                "message": f"잔고 조회 실패: {detail}",
            }

    # 5. 포트폴리오 상태
    if not user:
        portfolio = {
            "ok": None,
            "holding_source": None,
            "performance_source": None,
            "message": "login_required",
        }
    elif configured and account["ok"]:
        mode = settings.SYSTEM_KIS_MODE
        portfolio = {
            "ok": True,
            "holding_source": f"KIS {_ki_mode_label(mode)} 계좌",
            "performance_source": "앱 거래 기록 기준",
            "message": f"보유 현황은 KIS {_ki_mode_label(mode)} 계좌 기준입니다.",
        }
    else:
        portfolio = {
            "ok": False,
            "holding_source": "앱 DB",
            "performance_source": "앱 거래 기록 기준",
            "message": "KIS 계좌 미연결 — 앱 DB 기준으로 표시됩니다.",
        }

    # 6. AI 예측 상태
    # 우선순위: ai_service.get_prediction() 와 동일하게 유지
    #   1순위: DB 저장 예측 (predicted_prices IS NOT NULL)
    #   2순위: 로컬 LSTM 가중치 (.pth 파일)
    #   3순위: unavailable
    uploaded_count = await _count_uploaded_predictions(db)
    if uploaded_count > 0:
        ai = {
            "prediction_source": "uploaded",
            "message": f"업로드된 예측 데이터 {uploaded_count}건 사용",
        }
    else:
        try:
            from ml.predict import WEIGHTS_DIR  # noqa: PLC0415 — ML 모듈은 조건부 import 허용
            pth_count = len(list(WEIGHTS_DIR.glob("*.pth")))
        except Exception:
            pth_count = 0

        if pth_count > 0:
            ai = {
                "prediction_source": "local",
                "message": f"로컬 LSTM 가중치 {pth_count}종목 사용 가능",
            }
        else:
            ai = {
                "prediction_source": "unavailable",
                "message": "LSTM 가중치 없음 — AI 예측 비활성",
            }

    return {
        "backend": backend,
        "auth": auth,
        "kis": kis,
        "account": account,
        "portfolio": portfolio,
        "ai": ai,
        "checked_at": now.isoformat(),
    }
