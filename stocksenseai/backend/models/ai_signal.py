import uuid

from sqlalchemy import Column, DateTime, Numeric, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base


class AISignalHistory(Base):
    __tablename__ = "ai_signals_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    stock_code = Column(String(10), nullable=False, index=True)
    signal = Column(String(10), nullable=False)
    signal_score = Column(Numeric(5, 2))
    tech_score = Column(Numeric(5, 2))
    lstm_score = Column(Numeric(5, 2))
    rsi = Column(Numeric(8, 4))
    macd = Column(Numeric(12, 4))
    predicted_prices = Column(JSONB)
    confidence = Column(Numeric(5, 2))
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
