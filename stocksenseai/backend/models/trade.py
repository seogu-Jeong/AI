import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Numeric, String, func
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base


class Trade(Base):
    __tablename__ = "trades"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    stock_code = Column(String(10), nullable=False)
    stock_name = Column(String(100))
    order_type = Column(String(10), nullable=False)
    price_type = Column(String(10), nullable=False)
    quantity = Column(Integer, nullable=False)
    order_price = Column(Numeric(12, 2))
    executed_price = Column(Numeric(12, 2))
    commission = Column(Numeric(10, 2), server_default="0")
    realized_pnl = Column(Integer, nullable=True)
    filled_quantity = Column(Integer, nullable=False, default=0, server_default="0")
    status = Column(String(20), nullable=False, server_default="PENDING")
    mode = Column(String(20), nullable=False)
    kis_order_no = Column(String(50))
    ai_signal_at_order = Column(String(10))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    filled_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index("idx_trades_user_date", "user_id", "created_at"),
        Index("idx_trades_status", "user_id", "status", "mode"),
    )
