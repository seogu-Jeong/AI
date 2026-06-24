"""자동매매 Celery 태스크 — 장 중 10분마다 전체 활성 사용자 사이클 실행."""
import asyncio
import logging

from tasks import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="tasks.auto_trade_tasks.run_auto_trade_all")
def run_auto_trade_all() -> None:
    asyncio.run(_run_all_async())


async def _run_all_async() -> None:
    from sqlalchemy import select

    from core.database import AsyncSessionLocal
    from models.auto_trade import AutoTradeConfig
    from services.auto_trade_service import run_cycle

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(AutoTradeConfig).where(AutoTradeConfig.enabled.is_(True))
        )
        configs = result.scalars().all()

    for cfg in configs:
        try:
            async with AsyncSessionLocal() as db:
                result = await run_cycle(cfg.user_id, db)
                logger.info("자동매매 사이클 완료 user=%s executed=%s", cfg.user_id, result.get("executed", 0))
        except Exception as exc:
            logger.error("자동매매 사이클 실패 user=%s: %s", cfg.user_id, exc)
