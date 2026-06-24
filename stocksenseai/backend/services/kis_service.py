# backend/services/kis_service.py
from __future__ import annotations

import logging
from datetime import date as _date, datetime, timedelta, timezone

import httpx
from fastapi import HTTPException

_logger = logging.getLogger(__name__)
_KST = timezone(timedelta(hours=9))

from core.config import settings
from services.kis_token_service import get_access_token

_REAL_URL = "https://openapi.koreainvestment.com:9443"
_PAPER_URL = "https://openapivts.koreainvestment.com:29443"

_TR_IDS = {
    "buy":     {"real": "TTTC0802U", "paper": "VTTC0802U"},
    "sell":    {"real": "TTTC0801U", "paper": "VTTC0801U"},
    "cancel":  {"real": "TTTC0803U", "paper": "VTTC0803U"},
    "balance": {"real": "TTTC8434R", "paper": "VTTC8434R"},
    "fill":    {"real": "TTTC8001R", "paper": "VTTC8001R"},
}


def _base_url(mode: str) -> str:
    return _PAPER_URL if mode == "paper" else _REAL_URL


def _tr_id(action: str, mode: str) -> str:
    return _TR_IDS[action][mode]


def _get_keys(user=None, mode: str | None = None) -> tuple[str, str, str]:
    """.env SYSTEM_KIS 키 반환. (app_key, app_secret, account_no 하이픈 제거)"""
    app_key = settings.SYSTEM_KIS_APP_KEY
    app_secret = settings.SYSTEM_KIS_APP_SECRET
    account_no = settings.SYSTEM_KIS_ACCOUNT_NO.replace("-", "")

    if not app_key or not app_secret:
        raise HTTPException(status_code=503, detail="KIS API 키가 서버에 설정되지 않았습니다 (.env SYSTEM_KIS_APP_KEY)")
    if not account_no:
        raise HTTPException(status_code=503, detail="계좌번호가 서버에 설정되지 않았습니다 (.env SYSTEM_KIS_ACCOUNT_NO)")

    return app_key, app_secret, account_no


def _effective_mode(user=None, mode: str | None = None) -> str:
    """실제 사용할 KIS 모드 결정. 항상 .env SYSTEM_KIS_MODE 우선."""
    return settings.SYSTEM_KIS_MODE


async def _headers(user=None, mode: str | None = None) -> dict:
    app_key, app_secret, _ = _get_keys(user, mode)
    effective_mode = _effective_mode(user, mode)
    token = await get_access_token(app_key, app_secret, effective_mode)
    return {
        "authorization": f"Bearer {token}",
        "appkey": app_key,
        "appsecret": app_secret,
        "content-type": "application/json",
        "custtype": "P",
    }


async def place_order(
    user,
    stock_code: str,
    order_type: str,
    price_type: str,
    quantity: int,
    price: int = 0,
) -> dict:
    """
    KIS 주문 실행.
    order_type: "BUY" | "SELL"
    price_type: "MARKET" | "LIMIT"
    반환: {kis_order_no}
    """
    _, _, account_no = _get_keys(user)
    mode = _effective_mode(user)
    action = "buy" if order_type == "BUY" else "sell"
    ord_dvsn = "01" if price_type == "MARKET" else "00"

    headers = await _headers(user)
    headers["tr_id"] = _tr_id(action, mode)

    body = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "PDNO": stock_code,
        "ORD_DVSN": ord_dvsn,
        "ORD_QTY": str(quantity),
        "ORD_UNPR": str(price) if price_type == "LIMIT" else "0",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # hashkey 생성 후 헤더에 추가 (주문 변조 방지)
            hk_resp = await client.post(
                f"{_base_url(mode)}/uapi/hashkey",
                json=body,
                headers=headers,
            )
            if hk_resp.status_code == 200:
                headers["hashkey"] = hk_resp.json().get("HASH", "")

            resp = await client.post(
                f"{_base_url(mode)}/uapi/domestic-stock/v1/trading/order-cash",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        _logger.error("KIS 주문 실패 (status=%s): %s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=502, detail=f"KIS 주문 실패 (HTTP {exc.response.status_code})") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 연결 실패: {exc}") from exc

    if data.get("rt_cd") != "0":
        raise HTTPException(status_code=400, detail=f"KIS 주문 거부: {data.get('msg1', '')}")

    return {"kis_order_no": data["output"]["ODNO"]}


async def cancel_order(user, kis_order_no: str) -> dict:
    """미체결 주문 취소."""
    _, _, account_no = _get_keys(user)
    mode = _effective_mode(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("cancel", mode)

    body = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "KRX_FWDG_ORD_ORGNO": "",
        "ORGN_ODNO": kis_order_no,
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "02",
        "ORD_QTY": "0",
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # hashkey 생성 후 헤더에 추가 (주문 변조 방지)
            hk_resp = await client.post(
                f"{_base_url(mode)}/uapi/hashkey",
                json=body,
                headers=headers,
            )
            if hk_resp.status_code == 200:
                headers["hashkey"] = hk_resp.json().get("HASH", "")

            resp = await client.post(
                f"{_base_url(mode)}/uapi/domestic-stock/v1/trading/order-rvsecncl",
                json=body,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        _logger.error("KIS 취소 실패 (status=%s): %s", exc.response.status_code, exc.response.text)
        raise HTTPException(status_code=502, detail=f"KIS 취소 실패 (HTTP {exc.response.status_code})") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 취소 실패: {exc}") from exc

    if data.get("rt_cd") != "0":
        raise HTTPException(status_code=400, detail=f"KIS 취소 거부: {data.get('msg1', '')}")

    return {"cancelled": True}


async def poll_fill(user, kis_order_no: str, mode: str | None = None) -> dict | None:
    """
    체결 확인.
    반환: {executed_price, filled_qty, filled_at} or None (미체결)
    """
    mode = _effective_mode(user, mode)
    _, _, account_no = _get_keys(user, mode)
    headers = await _headers(user, mode)
    headers["tr_id"] = _tr_id("fill", mode)

    params = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "INQR_STRT_DT": "",
        "INQR_END_DT": "",
        "SLL_BUY_DVSN_CD": "00",
        "INQR_DVSN": "00",
        "PDNO": "",
        "CCLD_DVSN": "01",
        "ORD_GNO_BRNO": "",
        "ODNO": kis_order_no,
        "INQR_DVSN_3": "00",
        "INQR_DVSN_1": "",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base_url(mode)}/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"KIS 체결 조회 실패: {exc}") from exc

    _ensure_kis_ok(data, "체결 조회")
    output = data.get("output1", [])
    for item in output:
        if item.get("odno") == kis_order_no and item.get("ccld_dvsn") == "1":
            return {
                "executed_price": int(item.get("avg_prvs", 0)),
                "filled_qty": int(item.get("tot_ccld_qty", 0)),
                "filled_at": _parse_ord_tmd(item.get("ord_tmd", "")),
            }
    return None


def _ensure_kis_ok(data: dict, action: str) -> None:
    if data.get("rt_cd") != "0":
        raise HTTPException(status_code=502, detail=f"KIS {action} 오류: {data.get('msg1', '')}")


def _parse_ord_tmd(ord_tmd: str) -> datetime | None:
    """KIS ord_tmd (HHMMSS) → KST aware datetime (today 기준)."""
    if not ord_tmd or len(ord_tmd) < 6:
        return None
    try:
        today = _date.today()
        h, m, s = int(ord_tmd[0:2]), int(ord_tmd[2:4]), int(ord_tmd[4:6])
        return datetime(today.year, today.month, today.day, h, m, s, tzinfo=_KST)
    except (ValueError, IndexError):
        return None


async def get_balance(user) -> dict:
    """예수금 조회."""
    _, _, account_no = _get_keys(user)
    mode = _effective_mode(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("balance", mode)

    params = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base_url(mode)}/uapi/domestic-stock/v1/trading/inquire-balance",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"잔고 조회 실패: {exc}") from exc

    _ensure_kis_ok(data, "잔고 조회")
    output2 = data.get("output2", [{}])[0]
    return {
        "cash": int(output2.get("dnca_tot_amt", 0)),
        "total_eval": int(output2.get("tot_evlu_amt", 0)),
    }


async def get_balance_full(user) -> dict:
    """보유종목 리스트 + 예수금 전체 조회 (실거래 모드 포트폴리오용)."""
    _, _, account_no = _get_keys(user)
    mode = _effective_mode(user)
    headers = await _headers(user)
    headers["tr_id"] = _tr_id("balance", mode)

    params = {
        "CANO": account_no[:8],
        "ACNT_PRDT_CD": account_no[8:] if len(account_no) > 8 else "01",
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "00",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": "",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{_base_url(mode)}/uapi/domestic-stock/v1/trading/inquire-balance",
                params=params,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"잔고 조회 실패: {exc}") from exc

    _ensure_kis_ok(data, "잔고 조회")
    def _int(v: str) -> int:
        try:
            return int(float(v or 0))
        except (ValueError, TypeError):
            return 0

    def _float(v: str) -> float:
        try:
            return float(v or 0)
        except (ValueError, TypeError):
            return 0.0

    output1 = data.get("output1", [])
    output2 = data.get("output2", [{}])[0]

    holdings = [
        {
            "stock_code": item.get("pdno", ""),
            "stock_name": item.get("prdt_name", ""),
            "quantity": _int(item.get("hldg_qty", "0")),
            "avg_price": _float(item.get("pchs_avg_pric", "0")),
            "current_price": _int(item.get("prpr", "0")),
            "eval_amount": _int(item.get("evlu_amt", "0")),
            "profit_loss": _int(item.get("evlu_pfls_amt", "0")),
            "return_pct": round(_float(item.get("evlu_pfls_rt", "0")), 2),
        }
        for item in output1
        if _int(item.get("hldg_qty", "0")) > 0
    ]

    total_cost = sum(_int(item.get("pchs_amt", "0")) for item in output1 if _int(item.get("hldg_qty", "0")) > 0)

    return {
        "holdings": holdings,
        "total_eval": _int(output2.get("tot_evlu_amt", "0")),
        "total_cost": total_cost,
        "cash": _int(output2.get("dnca_tot_amt", "0")),
    }


async def test_kis_connection(app_key: str, app_secret: str, mode: str) -> bool:
    """KIS 연결 테스트 (기존 함수 유지)."""
    base_url = _PAPER_URL if mode == "paper" else _REAL_URL
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{base_url}/oauth2/tokenP",
                json={"grant_type": "client_credentials", "appkey": app_key, "appsecret": app_secret},
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code == 200 and bool(resp.json().get("access_token"))
        except Exception:
            return False
