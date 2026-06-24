import uuid

from sqlalchemy import Column, Date, DateTime, ForeignKey, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stock_code = Column(String(10))
    strategy_config = Column(JSONB, nullable=False)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    total_return_pct = Column(Numeric(10, 4))
    mdd_pct = Column(Numeric(10, 4))
    sharpe_ratio = Column(Numeric(8, 4))
    win_rate_pct = Column(Numeric(5, 2))
    total_trades = Column(Integer)
    result_detail = Column(JSONB)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
