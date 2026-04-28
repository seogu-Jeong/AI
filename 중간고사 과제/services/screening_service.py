import random
import pandas as pd
from datetime import datetime, date
from sqlalchemy import desc, func, and_
from db.models import AIScore, RawOHLCV, Feature

class ScreeningService:
    def get_top_n(self, n: int, db_session, signal_filter: str = None, search: str = None) -> list[dict]:
        # 1. Get latest date in AIScore
        latest_date_query = db_session.query(func.max(AIScore.date)).scalar()
        if not latest_date_query:
            return self.generate_mock_scores()[:n]

        # 2. Build query
        # Join AIScore with RawOHLCV to get close price
        # We need the latest price for each ticker. For simplicity, assume same date as AIScore or latest available.
        query = db_session.query(
            AIScore, 
            RawOHLCV.close
        ).join(
            RawOHLCV, 
            and_(AIScore.ticker == RawOHLCV.ticker, AIScore.date == RawOHLCV.date)
        ).filter(AIScore.date == latest_date_query)

        if signal_filter:
            query = query.filter(AIScore.mlp_signal == signal_filter)
        
        if search:
            query = query.filter(AIScore.ticker.like(f"%{search}%"))

        results = query.order_by(desc(AIScore.ensemble_score)).limit(n).all()

        if not results:
            return self.generate_mock_scores()[:n]

        # Optimization: Bulk-fetch all previous prices to avoid N+1 queries
        # 1. Get the previous trading date
        prev_date = db_session.query(func.max(RawOHLCV.date)).filter(
            RawOHLCV.date < latest_date_query
        ).scalar()

        # 2. Bulk-fetch all previous prices
        ticker_list = [score.ticker for score, _ in results]
        prev_price_map = {}
        if prev_date:
            prev_rows = db_session.query(RawOHLCV.ticker, RawOHLCV.close).filter(
                RawOHLCV.ticker.in_(ticker_list),
                RawOHLCV.date == prev_date
            ).all()
            prev_price_map = {r.ticker: r.close for r in prev_rows}

        output = []
        for i, (score, close_price) in enumerate(results):
            # Use bulk-fetched prev_price_map
            prev_close = prev_price_map.get(score.ticker, close_price)
            price_change_pct = (close_price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0

            score_val = round(score.ensemble_score, 2)
            derived_signal = 'Buy' if score_val >= 70 else ('Hold' if score_val >= 50 else 'Sell')

            output.append({
                "rank": i + 1,
                "ticker": score.ticker,
                "ensemble_score": score_val,
                "mlp_signal": derived_signal,
                "lstm_dir_5d": score.lstm_dir_5d,
                "cnn_pattern": score.cnn_pattern,
                "close_price": close_price,
                "price_change_pct": round(price_change_pct, 2)
            })

        return output
    def generate_mock_scores(self) -> list[dict]:
        tickers = ['NVDA','MSFT','AAPL','META','AMZN','GOOGL','BRK-B','LLY','AVGO','JPM','UNH','XOM','TSLA','PG','MA','JNJ','COST','HD','MRK','ABBV','CVX','BAC','NFLX','AMD','CRM','PEP','TMO','ADBE','ACN','LIN']
        scores = [87,81,76,72,68,65,83,71,69,78,74,62,45,70,77,68,73,66,64,69,61,59,71,63,75,67,72,70,68,65]
        
        mock_data = []
        for i, ticker in enumerate(tickers):
            # Simple deterministic hash for ticker
            h = sum(ord(c) * (j + 1) for j, c in enumerate(ticker))
            
            score = scores[i]
            signal = 'Buy' if score >= 70 else ('Hold' if score >= 50 else 'Sell')
            
            lstm_dirs = ['Up', 'Down', 'Side']
            cnn_patterns = ['Bullish Engulfing', 'Hammer', 'Doji', 'Bearish Engulfing']
            
            # Deterministic selections
            lstm_val = lstm_dirs[h % 3]
            cnn_val = cnn_patterns[(h // 3) % 4]
            
            # Price: deterministic based on score and hash
            # e.g. base price between 50 and 500
            base_price = 50 + (h % 450) + (h % 100) / 100.0
            
            # Change %: deterministic between -5 and +5
            change_pct = ((h % 1000) / 100.0) - 5.0
            
            mock_data.append({
                "rank": i + 1,
                "ticker": ticker,
                "ensemble_score": float(score),
                "mlp_signal": signal,
                "lstm_dir_5d": lstm_val,
                "cnn_pattern": cnn_val,
                "close_price": round(base_price, 2),
                "price_change_pct": round(change_pct, 2)
            })
        # Sort by score for consistency
        mock_data.sort(key=lambda x: x['ensemble_score'], reverse=True)
        for i, item in enumerate(mock_data):
            item['rank'] = i + 1
        return mock_data

    def get_ticker_detail(self, ticker: str, db_session) -> dict:
        latest_score = db_session.query(AIScore).filter(AIScore.ticker == ticker).order_by(desc(AIScore.date)).first()
        latest_feature = db_session.query(Feature).filter(Feature.ticker == ticker).order_by(desc(Feature.date)).first()
        latest_price = db_session.query(RawOHLCV).filter(RawOHLCV.ticker == ticker).order_by(desc(RawOHLCV.date)).first()

        if not latest_score or not latest_feature:
            import yfinance as yf
            try:
                hist = yf.download(ticker, period='6mo', auto_adjust=True, progress=False)
                if not hist.empty:
                    # flatten multi-index if present
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    mock_history = hist[['Open','High','Low','Close','Volume']].copy()
                else:
                    mock_history = None
            except Exception:
                mock_history = None

            # Return mock detail
            return {
                "ticker": ticker,
                "name": ticker,
                "date": date.today().isoformat(),
                "ensemble_score": 75.5,
                "mlp_signal": "Buy",
                "lstm_dir_5d": "Up",
                "lstm_dir_20d": "Up",
                "lstm_dir_60d": "Up",
                "cnn_pattern": "Hammer",
                "rsi_14": 62.1,
                "stoch_k": 65.0,
                "macd_signal": 0.45,
                "bb_pct": 0.78,
                "per": 28.5,
                "pbr": 4.2,
                "roe": 0.22,
                "eps_growth": 0.18,
                "op_margin": 0.25,
                "close_price": 150.25,
                "price": round(random.uniform(50, 900), 2),
                "price_change_pct": round(random.uniform(-3, 5), 2),
                "history": mock_history,
                "ai_score_history": [
                    {"date": date(2026, 1, 1), "ensemble": 60, "lstm": 55, "cnn": 65, "transformer": 62, "mlp": 58},
                    {"date": date(2026, 2, 1), "ensemble": 65, "lstm": 60, "cnn": 70, "transformer": 68, "mlp": 62},
                    {"date": date(2026, 3, 1), "ensemble": 72, "lstm": 68, "cnn": 75, "transformer": 74, "mlp": 70},
                    {"date": date(2026, 4, 1), "ensemble": 75.5, "lstm": 72, "cnn": 78, "transformer": 76, "mlp": 74},
                ],
                "attention_weights": None,
                "market_cap": 2.5e12,
                "dividend_yield": 1.2,
                "mlp_buy_prob": 0.72,
                "mlp_hold_prob": 0.20,
                "mlp_sell_prob": 0.08,
                "weights": [0.35, 0.25, 0.15, 0.25],
                "forecast": {
                    "dir_5d": "Up", "dir_20d": "Up", "dir_60d": "Up",
                    "mlp_buy_prob": 0.72, "mlp_sell_prob": 0.08,
                    "daily_std": 0.018, "current_price": 150.25
                }
            }

        # Fetch 10-year history: prefer yfinance (complete), fall back to DB
        df = None
        price_change_pct = 0.0
        try:
            import yfinance as yf
            _hist = yf.download(ticker, period='10y', auto_adjust=True, progress=False)
            if not _hist.empty:
                if isinstance(_hist.columns, pd.MultiIndex):
                    _hist.columns = _hist.columns.get_level_values(0)
                if hasattr(_hist.index, 'tz') and _hist.index.tz is not None:
                    _hist.index = _hist.index.tz_localize(None)
                df = _hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                if len(_hist) >= 2:
                    close_now = float(_hist['Close'].iloc[-1])
                    close_prev = float(_hist['Close'].iloc[-2])
                    if close_prev != 0:
                        price_change_pct = (close_now - close_prev) / close_prev * 100
        except Exception as e:
            print(f"yfinance 10y fetch failed for {ticker}: {e}")

        # Fall back to DB if yfinance failed
        if df is None:
            history_rows = db_session.query(RawOHLCV).filter(
                RawOHLCV.ticker == ticker
            ).order_by(desc(RawOHLCV.date)).limit(2520).all()
            if history_rows:
                data_rows = []
                for r in history_rows:
                    data_rows.append([r.date, r.open, r.high, r.low, r.close, r.volume])
                df = pd.DataFrame(data_rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df = df.sort_index()
                if len(history_rows) >= 2:
                    close_now = history_rows[0].close
                    close_prev = history_rows[1].close
                    if close_prev != 0:
                        price_change_pct = (close_now - close_prev) / close_prev * 100

        # Get latest weights or defaults
        from db.models import EnsembleWeight
        w = db_session.query(EnsembleWeight).order_by(desc(EnsembleWeight.date)).first()
        weights = [w.lstm_w, w.cnn_w, w.transformer_w, w.mlp_w] if w else [0.25, 0.25, 0.25, 0.25]

        # Calculate daily_std for forecast
        daily_std = 0.02 # default 2%
        if df is not None and len(df) >= 2:
            returns = df['Close'].pct_change().dropna()
            if not returns.empty:
                daily_std = float(returns.tail(60).std())

        current_price = latest_price.close if latest_price else (df['Close'].iloc[-1] if df is not None and not df.empty else 0)

        forecast_data = {
            'dir_5d': latest_score.lstm_dir_5d,
            'dir_20d': latest_score.lstm_dir_20d,
            'dir_60d': latest_score.lstm_dir_60d,
            'mlp_buy_prob': latest_score.mlp_buy_prob or 0,
            'mlp_sell_prob': latest_score.mlp_sell_prob or 0,
            'daily_std': daily_std,
            'current_price': current_price
        }

        # AI Score History: DB-level limit for performance
        score_history_rows = db_session.query(AIScore)\
            .filter(AIScore.ticker == ticker)\
            .order_by(desc(AIScore.date))\
            .limit(24)\
            .all()
        score_history_rows = list(reversed(score_history_rows))

        # Remove isolated outlier dates caused by large gaps (e.g. today's inference
        # score 2026-04-24 isolated from last monthly backfill date 2023-08-01)
        if len(score_history_rows) > 1:
            filtered = [score_history_rows[0]]
            for i in range(1, len(score_history_rows)):
                days_gap = (score_history_rows[i].date - score_history_rows[i-1].date).days
                if days_gap <= 400:
                    filtered.append(score_history_rows[i])
            score_history_rows = filtered

        score_history = []
        for r in score_history_rows:
            score_history.append({
                "date": r.date,
                "ensemble": r.ensemble_score,
                "lstm": r.lstm_score,
                "cnn": r.cnn_score,
                "transformer": r.transformer_score,
                "mlp": r.mlp_score
            })

        return {
            "ticker": latest_score.ticker,
            "name": '',
            "date": latest_score.date.isoformat(),
            "ensemble_score": latest_score.ensemble_score,
            "mlp_score": latest_score.mlp_score,
            "lstm_score": latest_score.lstm_score,
            "cnn_score": latest_score.cnn_score,
            "transformer_score": latest_score.transformer_score,
            "mlp_signal": latest_score.mlp_signal,
            "lstm_dir_5d": latest_score.lstm_dir_5d,
            "lstm_dir_20d": latest_score.lstm_dir_20d,
            "lstm_dir_60d": latest_score.lstm_dir_60d,
            "cnn_pattern": latest_score.cnn_pattern,
            "rsi_14": latest_feature.rsi_14,
            "stoch_k": latest_feature.stoch_k,
            "macd_signal": latest_feature.macd_signal,
            "bb_pct": latest_feature.bb_pct,
            "per": latest_feature.per,
            "pbr": latest_feature.pbr,
            "roe": latest_feature.roe,
            "eps_growth": latest_feature.eps_growth,
            "op_margin": latest_feature.op_margin,
            "close_price": current_price,
            "price": current_price,
            "price_change_pct": round(price_change_pct, 2),
            "history": df,
            "ai_score_history": score_history,
            "attention_weights": None,
            "market_cap": latest_feature.market_cap,
            "dividend_yield": latest_feature.dividend_yield,
            "mlp_buy_prob": latest_score.mlp_buy_prob or 0,
            "mlp_hold_prob": latest_score.mlp_hold_prob or 0,
            "mlp_sell_prob": latest_score.mlp_sell_prob or 0,
            "weights": weights,
            "forecast": forecast_data
        }

