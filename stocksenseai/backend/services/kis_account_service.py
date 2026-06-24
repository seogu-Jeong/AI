# backend/services/kis_account_service.py
import logging

import httpx
from fastapi import HTTPException

from core.config import settings
from services.kis_token_service import get_access_token

_logger = logging.getLogger(__name__)

_REAL_URL = "https://openapi.koreainvestment.com:9443"
_PAPER_URL = "https://openapivts.koreainvestment.com:29443"

_BALANCE_TR = {"real": "TTTC8434R", "paper": "VTTC8434R"}


def _base(mode: str) -> str:
    return _PAPER_URL if mode == "paper" else _REAL_URL


def _parse_account_no(raw: str) -> tuple[str, str]:
    """'12345678-01' → (cano 8자리, prod 2자리)"""
    clean = raw.replace("-", "")
    return clean[:8], clean[8:] or "01"


def mask_account_no(raw: str) -> str:
    if not raw:
        return ""
    cano, prod = _parse_account_no(raw)
    return f"{cano[:4]}****-{prod}"


async def get_account_balance(mode: str) -> dict:
    """KIS 잔고조회 → {summary, holdings} 반환. .env SYSTEM_KIS 키 사용."""
    app_key = settings.SYSTEM_KIS_APP_KEY
    app_secret = settings.SYSTEM_KIS_APP_SECRET
    account_no_raw = settings.SYSTEM_KIS_ACCOUNT_NO

    if not app_key or not app_secret:
        raise HTTPException(status_code=503, detail="KIS API 키가 서버에 설정되지 않았습니다 (.env SYSTEM_KIS_APP_KEY)")
    if not account_no_raw:
        raise HTTPException(status_code=503, detail="계좌번호가 서버에 설정되지 않았습니다 (.env SYSTEM_KIS_ACCOUNT_NO)")

    cano, prod = _parse_account_no(account_no_raw)
    token = await get_access_token(app_key, app_secret, mode)
    tr_id = _BALANCE_TR[mode]

    headers = {
        "Authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "tr_id": tr_id,
        "custtype": "P",
        "Content-Type": "application/json; charset=utf-8",
    }
    params = {
        "CANO": cano,
        "ACNT_PRDT_CD": prod,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "00",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "00",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base(mode)}/uapi/domestic-stock/v1/trading/inquire-balance",
                headers=headers,
                params=params,
            )
            resp.raise_for_status()
            body = resp.json()
    except httpx.HTTPStatusError as exc:
        _logger.error("KIS 잔고 조회 실패 (status=%s): %s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=502, detail=f"KIS 잔고 조회 실패 (HTTP {exc.response.status_code})") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 연결 실패: {exc}") from exc

    if body.get("rt_cd") != "0":
        raise HTTPException(status_code=502, detail=f"KIS 오류: {body.get('msg1', '')}")

    # output2: 계좌 요약
    o2 = body.get("output2", [{}])
    o2 = o2[0] if o2 else {}
    deposit = int(float(o2.get("dnca_tot_amt", 0)))
    eval_amt = int(float(o2.get("scts_evlu_amt", 0)))
    buy_amt = int(float(o2.get("pchs_amt_smtl_amt", 0)))
    eval_pfls = int(float(o2.get("evlu_pfls_smtl_amt", 0)))
    total_asset = deposit + eval_amt

    summary = {
        "total_asset": total_asset,
        "deposit": deposit,
        "eval_amount": eval_amt,
        "buy_amount": buy_amt,
        "eval_profit_loss": eval_pfls,
        "return_pct": round(eval_pfls / buy_amt * 100, 2) if buy_amt else 0.0,
    }

    # output1: 종목별 보유 내역
    holdings = []
    for item in body.get("output1", []):
        qty = int(item.get("hldg_qty", 0))
        if qty == 0:
            continue
        holdings.append({
            "stock_code": item.get("pdno", ""),
            "stock_name": item.get("prdt_name", ""),
            "quantity": qty,
            "avg_price": int(float(item.get("pchs_avg_pric", 0))),
            "current_price": int(float(item.get("prpr", 0))),
            "eval_amount": int(float(item.get("evlu_amt", 0))),
            "profit_loss": int(float(item.get("evlu_pfls_amt", 0))),
            "return_pct": float(item.get("evlu_erng_rt", 0)),
        })

    return {
        "mode": mode,
        "account_no": mask_account_no(account_no_raw),
        "summary": summary,
        "holdings": holdings,
        "data_source": f"KIS {'모의투자' if mode == 'paper' else '실계좌'} 계좌",
    }
