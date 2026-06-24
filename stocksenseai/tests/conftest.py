import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

os.environ["APP_ENV"] = "test"


def _test_database_url() -> str:
    explicit = os.environ.get("TEST_DATABASE_URL")
    source = explicit or os.environ.get(
        "DATABASE_URL",
        "postgresql+asyncpg://stocksense:stocksense@localhost:5432/stocksense_test",
    )
    url = make_url(source)
    database = url.database or "stocksense"
    if not database.endswith("_test"):
        url = url.set(database=f"{database}_test")
    if not (url.database or "").endswith("_test"):
        raise RuntimeError(f"Refusing to run tests against non-test database: {url.database}")
    return url.render_as_string(hide_password=False)


TEST_DB_URL = _test_database_url()
os.environ["DATABASE_URL"] = TEST_DB_URL

from api.middleware.rate_limit import limiter  # noqa: E402
from core.database import Base, get_db  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the in-memory rate limiter before each test to prevent cross-test pollution."""
    limiter.reset()
    yield


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="session")
async def db_session(test_engine):
    conn = await test_engine.connect()
    trans = await conn.begin()
    session = AsyncSession(bind=conn, expire_on_commit=False)

    yield session

    await session.close()
    await trans.rollback()
    await conn.close()


@pytest_asyncio.fixture(loop_scope="session")
async def client(db_session):
    async def override_db():
        yield db_session

    app.dependency_overrides[get_db] = override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="https://test") as ac:
        yield ac
    app.dependency_overrides.clear()
