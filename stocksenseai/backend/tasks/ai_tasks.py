import asyncio
import logging

from tasks import celery_app

logger = logging.getLogger(__name__)


@celery_app.task
def refresh_ai_signals() -> None:
    """장 종료 후(15:35) 학습된 전 종목 AI 시그널 갱신."""
    asyncio.run(_refresh_async())


async def _refresh_async() -> None:
    from core.database import AsyncSessionLocal
    from ml.predict import WEIGHTS_DIR
    from services.ai_service import calculate_signal

    codes = [p.stem for p in WEIGHTS_DIR.glob("*.pth")]
    success, failed = 0, []
    for code in codes:
        try:
            async with AsyncSessionLocal() as db:
                await calculate_signal(code, db)
            success += 1
        except Exception as e:
            logger.error("[AI 갱신 실패] %s: %s", code, e)
            failed.append(code)

    logger.info("[AI 갱신 완료] 성공=%d 실패=%d %s", success, len(failed), failed or "")
