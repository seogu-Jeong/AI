from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.middleware.sessions import SessionMiddleware

from api.middleware.rate_limit import limiter
from api.routes import account as account_router
from api.routes import ai as ai_router
from api.routes import alerts as alerts_router
from api.routes import analysis as analysis_router
from api.routes import auth as auth_router
from api.routes import backtest as backtest_router
from api.routes import portfolio as portfolio_router
from api.routes import realtime as realtime_router
from api.routes import risk as risk_router
from api.routes import simulate as simulate_router
from api.routes import stocks as stocks_router
from api.routes import auto_trade as auto_trade_router
from api.routes import system as system_router
from api.routes import trades as trades_router
from api.routes import watchlist as watchlist_router
from core.config import settings
from core.redis_client import close_redis
from services.websocket_service import kis_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await kis_pool.stop()
    await close_redis()


app = FastAPI(title="StockSenseAI API", lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(stocks_router.router, prefix="/stocks", tags=["stocks"])
app.include_router(realtime_router.router, tags=["realtime"])
app.include_router(ai_router.router, prefix="/ai", tags=["ai"])
app.include_router(trades_router.router, prefix="/trades", tags=["trades"])
app.include_router(portfolio_router.router, prefix="/portfolio", tags=["portfolio"])
app.include_router(risk_router.router, prefix="/risk", tags=["risk"])
app.include_router(alerts_router.router, prefix="/alerts", tags=["alerts"])
app.include_router(backtest_router.router, prefix="/backtest", tags=["backtest"])
app.include_router(simulate_router.router, prefix="/simulate", tags=["simulate"])
app.include_router(watchlist_router.router, prefix="/watchlist", tags=["watchlist"])
app.include_router(account_router.router, prefix="/account", tags=["account"])
app.include_router(analysis_router.router, prefix="/analysis", tags=["analysis"])
app.include_router(system_router.router, prefix="/system", tags=["system"])
app.include_router(auto_trade_router.router, prefix="/auto-trade", tags=["auto-trade"])


@app.get("/health")
async def health():
    return {"status": "ok"}
