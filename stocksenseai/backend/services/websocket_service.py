# backend/services/websocket_service.py
import asyncio
import json
import logging
import math
from collections import defaultdict

import websockets

from core.config import settings
from core.redis_client import get_redis
from services.kis_token_service import get_approval_key

logger = logging.getLogger(__name__)

_KIS_WS_PAPER = "ws://ops.koreainvestment.com:31000/"
_KIS_WS_REAL = "ws://ops.koreainvestment.com:21000/"

MAX_PER_SESSION = 41


def _ws_url(mode: str) -> str:
    return _KIS_WS_PAPER if mode == "paper" else _KIS_WS_REAL


def _parse_execution_msg(raw: str) -> dict | None:
    """H0STCNT0 pipe-caret format parser. Returns None for non-data messages."""
    if raw.startswith("1"):
        return None
    parts = raw.split("|")
    if len(parts) < 4 or parts[1] != "H0STCNT0":
        return None
    fields = parts[3].split("^")
    if len(fields) < 13:
        return None
    return {
        "type": "execution",
        "code": fields[0],
        "time": fields[1],
        "price": int(fields[2]) if fields[2].lstrip("-").isdigit() else 0,
        "sign": fields[3],
        "change": int(fields[4]) if fields[4].lstrip("-").isdigit() else 0,
        "change_rate": float(fields[5]) if fields[5] else 0.0,
        "volume": int(fields[12]) if fields[12].lstrip("-").isdigit() else 0,
    }


class KISWebSocketPool:
    def __init__(self) -> None:
        self._subscriptions: dict[str, int] = defaultdict(int)
        self._sessions: list = []
        self._symbol_session: dict[str, int] = {}

    def subscription_count(self, code: str) -> int:
        return self._subscriptions.get(code, 0)

    async def subscribe(self, code: str) -> None:
        if not settings.SYSTEM_KIS_APP_KEY:
            return
        self._subscriptions[code] += 1
        if self._subscriptions[code] == 1:
            session = await self._get_session()
            if session:
                self._symbol_session[code] = self._sessions.index(session)
                await self._send_subscribe(code, tr_type="1", session=session)

    async def unsubscribe(self, code: str) -> None:
        if not settings.SYSTEM_KIS_APP_KEY:
            return
        if self._subscriptions.get(code, 0) <= 0:
            return
        self._subscriptions[code] -= 1
        if self._subscriptions[code] == 0:
            idx = self._symbol_session.pop(code, None)
            if idx is not None and idx < len(self._sessions) and not self._sessions[idx].closed:
                await self._send_subscribe(code, tr_type="2", session=self._sessions[idx])

    async def _send_subscribe(self, code: str, tr_type: str, session=None) -> None:
        approval_key = await get_approval_key(
            settings.SYSTEM_KIS_APP_KEY,
            settings.SYSTEM_KIS_APP_SECRET,
            settings.SYSTEM_KIS_MODE,
        )
        msg = {
            "header": {
                "approval_key": approval_key,
                "custtype": "P",
                "tr_type": tr_type,
                "content-type": "utf-8",
            },
            "body": {"input": {"tr_id": "H0STCNT0", "tr_key": code}},
        }
        target = session if session is not None else await self._get_session()
        if target:
            await target.send(json.dumps(msg))

    async def _get_session(self):
        active = [s for s in self._sessions if not s.closed]
        subscribed_count = sum(1 for v in self._subscriptions.values() if v > 0)
        needed = max(1, math.ceil(subscribed_count / MAX_PER_SESSION))

        while len(active) < needed:
            try:
                ws = await websockets.connect(
                    _ws_url(settings.SYSTEM_KIS_MODE),
                    ping_interval=30,
                    ping_timeout=10,
                )
                active.append(ws)
                self._sessions = active
                asyncio.create_task(self._recv_loop(ws))
            except Exception as e:
                logger.warning("KIS WS connection failed: %s", e)
                return None

        self._sessions = active
        return active[-1] if active else None

    async def _recv_loop(self, ws) -> None:
        try:
            async for raw in ws:
                await self._on_raw_message(str(raw))
        except Exception as e:
            logger.debug("KIS WS recv_loop ended: %s", e)

    async def _on_raw_message(self, raw: str) -> None:
        data = _parse_execution_msg(raw)
        if data:
            redis = await get_redis()
            await redis.publish(f"stock:{data['code']}", json.dumps(data))

    async def stop(self) -> None:
        for ws in self._sessions:
            try:
                await ws.close()
            except Exception:
                pass
        self._sessions.clear()


kis_pool = KISWebSocketPool()
