import uuid

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base


class RiskSettings(Base):
    __tablename__ = "risk_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    max_per_stock_pct = Column(Numeric(5, 2), server_default="20.0")
    daily_loss_limit_pct = Column(Numeric(5, 2), server_default="5.0")
    stop_loss_enabled = Column(Boolean, server_default="false")
    trading_blocked = Column(Boolean, server_default="false")
    enforce_hard_stop = Column(Boolean, server_default="true")
    blocked_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AlertSettings(Base):
    __tablename__ = "alert_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    signal_change = Column(Boolean, server_default="true")
    watchlist_price = Column(Boolean, server_default="true")
    daily_loss_limit = Column(Boolean, server_default="true")
    trade_filled = Column(Boolean, server_default="true")
    weekly_report = Column(Boolean, server_default="false")
    notification_email = Column(String(255), nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
