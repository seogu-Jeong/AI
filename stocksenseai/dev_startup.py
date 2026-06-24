"""로컬 개발용 시작 스크립트 — PostgreSQL/Redis 없이 SQLite + fakeredis로 구동."""
import os
import sys

# 반드시 모든 import 전에 환경변수 설정
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./stocksense_dev.db"
os.environ["APP_ENV"] = "development"
os.environ.setdefault("SECRET_KEY", "dev-secret-key-change-in-production-minimum32chars")
os.environ.setdefault("ENCRYPTION_KEY", "L+PBIjBrV+1BIZLcNKVm/sGtjWMomsJXMtRVcSnyrIU=")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")

# PostgreSQL UUID / JSONB → SQLite 호환 타입으로 교체
import uuid as _uuid
from sqlalchemy import types as _sa_types
from sqlalchemy.dialects import postgresql as _pg

class _SQLiteUUID(_sa_types.TypeDecorator):
    impl = _sa_types.String(36)
    cache_ok = True

    def __init__(self, as_uuid=True, **kwargs):
        self.as_uuid = as_uuid
        super().__init__(**kwargs)

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return _uuid.UUID(value) if self.as_uuid else value

_pg.UUID = _SQLiteUUID  # type: ignore
_pg.JSONB = _sa_types.JSON  # type: ignore

# fakeredis 패치
import fakeredis.aioredis as _fakeredis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

# redis_client 모듈을 fakeredis로 교체
import types as _types
_redis_mod = _types.ModuleType("core.redis_client")
_fake_instance = None

async def _get_redis():
    global _fake_instance
    if _fake_instance is None:
        _fake_instance = _fakeredis.FakeRedis(decode_responses=True)
    return _fake_instance

async def _close_redis():
    global _fake_instance
    _fake_instance = None

_redis_mod.get_redis = _get_redis
_redis_mod.close_redis = _close_redis
sys.modules["core.redis_client"] = _redis_mod

# SQLite에서 create_all로 테이블 생성 (Alembic 대신)
async def _init_db():
    from sqlalchemy.ext.asyncio import create_async_engine
    from core.database import Base
    import models.user       # noqa
    import models.ai_signal  # noqa
    import models.backtest   # noqa
    import models.portfolio  # noqa
    import models.price_cache # noqa
    import models.risk       # noqa
    import models.trade      # noqa
    import models.watchlist  # noqa
    import models.auto_trade  # noqa

    engine = create_async_engine(os.environ["DATABASE_URL"], echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("✅ SQLite 테이블 생성 완료")

import asyncio
asyncio.run(_init_db())

# uvicorn으로 앱 실행
import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
