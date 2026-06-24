from sqlalchemy import Column, Date, Index, Numeric, String

from core.database import Base


class PriceCache(Base):
    __tablename__ = "price_cache"

    ticker = Column(String(20), primary_key=True, nullable=False)
    trade_date = Column(Date, primary_key=True, nullable=False)
    close_price = Column(Numeric(12, 2), nullable=False)
    market = Column(String(10), primary_key=True, nullable=False, default="KR")

    __table_args__ = (
        Index("idx_price_cache_ticker_market", "ticker", "market"),
    )
