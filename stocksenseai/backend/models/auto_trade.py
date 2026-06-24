import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base


class AutoTradeConfig(Base):
    __tablename__ = "auto_trade_configs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)

    enabled = Column(Boolean, nullable=False, default=False, server_default="false")
    mode = Column(String(10), nullable=False, default="paper", server_default="paper")

    total_budget = Column(Integer, nullable=False, default=1000000)
    budget_per_trade = Column(Integer, nullable=False, default=100000)
    max_positions = Column(Integer, nullable=False, default=5)
    signal_threshold = Column(Integer, nullable=False, default=70)
    stop_loss_pct = Column(Float, nullable=False, default=5.0)
    take_profit_pct = Column(Float, nullable=False, default=10.0)
    watch_codes = Column(JSONB, nullable=False, default=list)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_auto_trade_config_user", "user_id"),)


class AutoTradeLog(Base):
    __tablename__ = "auto_trade_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(100))
    action = Column(String(20), nullable=False)
    quantity = Column(Integer, nullable=False, default=0)
    price = Column(Integer, nullable=False, default=0)
    total_amount = Column(Integer, nullable=False, default=0)
    reason = Column(String(200))
    signal_score = Column(Float, default=0.0)
    mode = Column(String(10), nullable=False, default="paper")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (Index("idx_auto_trade_logs_user", "user_id", "created_at"),)
