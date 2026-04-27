from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, UniqueConstraint
from sqlalchemy.sql import func
from db.database import Base

class RawOHLCV(Base):
    __tablename__ = "raw_ohlcv"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    adj_close = Column(Float)
    dividend = Column(Float, default=0.0)
    __table_args__ = (UniqueConstraint('ticker', 'date', name='_ticker_date_uc'),)

class Feature(Base):
    __tablename__ = "features"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, index=True)
    date = Column(Date, index=True)
    # 24 Feature columns
    rsi_14 = Column(Float)
    macd_signal = Column(Float)
    bb_pct = Column(Float)
    atr_14 = Column(Float)
    obv_chg = Column(Float)
    vol_ratio = Column(Float)
    stoch_k = Column(Float)
    ma_cross = Column(Float)
    per = Column(Float)
    pbr = Column(Float)
    roe = Column(Float)
    eps_growth = Column(Float)
    debt_ratio = Column(Float)
    op_margin = Column(Float)
    market_cap = Column(Float)
    high52w_ratio = Column(Float)
    ret_1m = Column(Float)
    ret_3m = Column(Float)
    usdkrw_chg = Column(Float)
    vix = Column(Float)
    news_sentiment = Column(Float)
    news_sentiment_5d = Column(Float)
    dividend_yield = Column(Float)
    eps_surprise = Column(Float)
    __table_args__ = (UniqueConstraint('ticker', 'date', name='_ticker_date_feat_uc'),)

class AIScore(Base):
    __tablename__ = "ai_scores"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, index=True)
    date = Column(Date, index=True)
    lstm_score = Column(Float)
    cnn_score = Column(Float)
    transformer_score = Column(Float)
    mlp_score = Column(Float)
    ensemble_score = Column(Float)
    lstm_dir_5d = Column(Text)
    lstm_dir_20d = Column(Text)
    lstm_dir_60d = Column(Text)
    cnn_pattern = Column(Text)
    mlp_signal = Column(Text)
    mlp_buy_prob = Column(Float)
    mlp_hold_prob = Column(Float)
    mlp_sell_prob = Column(Float)
    __table_args__ = (UniqueConstraint('ticker', 'date', name='_ticker_date_score_uc'),)

class EnsembleWeight(Base):
    __tablename__ = "ensemble_weights"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    lstm_w = Column(Float)
    cnn_w = Column(Float)
    transformer_w = Column(Float)
    mlp_w = Column(Float)
    lstm_acc = Column(Float)
    cnn_acc = Column(Float)
    trans_acc = Column(Float)
    mlp_acc = Column(Float)

class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, index=True)
    date = Column(Date, index=True)
    headline = Column(Text)
    source = Column(Text)
    score = Column(Float)
    label = Column(Text)

class Watchlist(Base):
    __tablename__ = "watchlist"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, unique=True, index=True)
    added_at = Column(DateTime, server_default=func.now())
    alert_threshold = Column(Float, default=15.0)

class AlertHistory(Base):
    __tablename__ = "alert_history"
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(Text, index=True)
    fired_at = Column(DateTime, server_default=func.now())
    score_before = Column(Float)
    score_after = Column(Float)
    delta = Column(Float)

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(Text)
    risk_level = Column(Text)
    created_at = Column(DateTime, server_default=func.now())
    holdings = Column(Text)  # Stores JSON string

class PaperAccount(Base):
    __tablename__ = 'paper_account'
    id = Column(Integer, primary_key=True)
    initial_cash = Column(Float, default=100000.0)
    current_cash = Column(Float, default=100000.0)
    created_at = Column(DateTime, server_default=func.now())

class PaperPosition(Base):
    __tablename__ = 'paper_positions'
    id = Column(Integer, primary_key=True)
    ticker = Column(Text, index=True)
    quantity = Column(Float)
    avg_cost = Column(Float)
    updated_at = Column(DateTime, server_default=func.now())
    __table_args__ = (UniqueConstraint('ticker', name='_paper_pos_ticker_uc'),)

class PaperTransaction(Base):
    __tablename__ = 'paper_transactions'
    id = Column(Integer, primary_key=True)
    action = Column(Text) # BUY or SELL
    ticker = Column(Text, index=True)
    quantity = Column(Float)
    price = Column(Float)
    total = Column(Float)
    timestamp = Column(DateTime, server_default=func.now())
