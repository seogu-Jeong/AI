import pandas as pd
import numpy as np
import yfinance as yf
import math
from ta.momentum import RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from db.models import RawOHLCV, Feature
from sqlalchemy.orm import Session
from sqlalchemy import select

class Preprocessor:
    def __init__(self):
        self._fund_cache = {}

    def _fetch_fundamental(self, ticker: str) -> dict:
        if ticker in self._fund_cache:
            return self._fund_cache[ticker]
        
        try:
            info = yf.Ticker(ticker).info
            mcap = info.get('marketCap', 0)
            if mcap and mcap > 0:
                mcap = math.log10(mcap)
            else:
                mcap = 0.0

            data = {
                'per': float(info.get('trailingPE', 0.0) or 0.0),
                'pbr': float(info.get('priceToBook', 0.0) or 0.0),
                'roe': float(info.get('returnOnEquity', 0.0) or 0.0),
                'eps_growth': float(info.get('earningsGrowth', 0.0) or 0.0),
                'debt_ratio': float(info.get('debtToEquity', 0.0) or 0.0),
                'op_margin': float(info.get('operatingMargins', 0.0) or 0.0),
                'market_cap': float(mcap),
                'dividend_yield': float(info.get('dividendYield', 0.0) or 0.0),
                'eps_surprise': float(info.get('surpriseEarningsPerShare', 0.0) or 0.0)
            }
        except Exception:
            data = {
                'per': 0.0, 'pbr': 0.0, 'roe': 0.0, 'eps_growth': 0.0,
                'debt_ratio': 0.0, 'op_margin': 0.0, 'market_cap': 0.0,
                'dividend_yield': 0.0, 'eps_surprise': 0.0
            }
        
        self._fund_cache[ticker] = data
        return data

    def _fetch_macro(self) -> dict:
        try:
            vix_ticker = yf.Ticker('^VIX')
            vix_hist = vix_ticker.history(period='2d')
            vix = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 0.0

            krw_ticker = yf.Ticker('KRW=X')
            krw_hist = krw_ticker.history(period='2d')
            if len(krw_hist) >= 2:
                yesterday = krw_hist['Close'].iloc[-2]
                today = krw_hist['Close'].iloc[-1]
                usdkrw_chg = float((today - yesterday) / yesterday)
            else:
                usdkrw_chg = 0.0
            
            return {'vix': vix, 'usdkrw_chg': usdkrw_chg}
        except Exception:
            return {'vix': 0.0, 'usdkrw_chg': 0.0}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure df has enough data
        if len(df) < 30:
            return df

        # RSI
        df['rsi_14'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd_signal'] = macd.macd_diff() # Using macd_diff as the "signal" component often used in features, or just macd_signal
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_pct'] = bb.bollinger_pband()
        
        # ATR
        df['atr_14'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # OBV
        obv_indicator = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv_indicator.on_balance_volume()
        df['obv_chg'] = df['obv'].pct_change()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
        df['stoch_k'] = stoch.stoch()
        
        # EMA
        ema5 = EMAIndicator(close=df['close'], window=5).ema_indicator()
        ema20 = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema5'] = ema5
        df['ema20'] = ema20
        
        # Custom logic
        df['ma_cross'] = np.where(df['ema5'] > df['ema20'], 1.0, -1.0)
        df['vol_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Select final columns
        return df

    def compute_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ret_1m'] = df['close'].pct_change(21)
        df['ret_3m'] = df['close'].pct_change(63)
        df['high52w_ratio'] = df['close'] / df['close'].rolling(window=252, min_periods=1).max()
        return df

    def build_features(self, ticker: str, db_session: Session) -> pd.DataFrame:
        stmt = select(RawOHLCV).where(RawOHLCV.ticker == ticker).order_by(RawOHLCV.date)
        rows = db_session.execute(stmt).scalars().all()
        
        if not rows:
            return pd.DataFrame()
        
        data = [{
            'date': r.date,
            'open': r.open,
            'high': r.high,
            'low': r.low,
            'close': r.close,
            'volume': r.volume
        } for r in rows]
        
        df = pd.DataFrame(data)
        df = self.compute_indicators(df)
        df = self.compute_momentum(df)
        
        # Real data fetching
        fund = self._fetch_fundamental(ticker)
        macro = self._fetch_macro()
        
        # Populate columns
        df['per'] = fund['per']
        df['pbr'] = fund['pbr']
        df['roe'] = fund['roe']
        df['eps_growth'] = fund['eps_growth']
        df['debt_ratio'] = fund['debt_ratio']
        df['op_margin'] = fund['op_margin']
        df['market_cap'] = fund['market_cap']
        df['dividend_yield'] = fund['dividend_yield']
        df['eps_surprise'] = fund['eps_surprise']
        df['vix'] = macro['vix']
        df['usdkrw_chg'] = macro['usdkrw_chg']
        df['news_sentiment'] = 0.0
        df['news_sentiment_5d'] = 0.0
            
        # Drop rows with NaN if any (from rolling/indicators)
        df = df.dropna()
        
        placeholders = [
            'per', 'pbr', 'roe', 'eps_growth', 'debt_ratio', 'op_margin', 
            'market_cap', 'usdkrw_chg', 'vix', 'news_sentiment', 
            'news_sentiment_5d', 'dividend_yield', 'eps_surprise'
        ]

        for _, row in df.iterrows():
            stmt = sqlite_insert(Feature).values(
                ticker=ticker,
                date=row['date'],
                rsi_14=float(row['rsi_14']),
                macd_signal=float(row['macd_signal']),
                bb_pct=float(row['bb_pct']),
                atr_14=float(row['atr_14']),
                obv_chg=float(row['obv_chg']),
                vol_ratio=float(row['vol_ratio']),
                stoch_k=float(row['stoch_k']),
                ma_cross=float(row['ma_cross']),
                per=float(row['per']),
                pbr=float(row['pbr']),
                roe=float(row['roe']),
                eps_growth=float(row['eps_growth']),
                debt_ratio=float(row['debt_ratio']),
                op_margin=float(row['op_margin']),
                market_cap=float(row['market_cap']),
                high52w_ratio=float(row['high52w_ratio']),
                ret_1m=float(row['ret_1m']),
                ret_3m=float(row['ret_3m']),
                usdkrw_chg=float(row['usdkrw_chg']),
                vix=float(row['vix']),
                news_sentiment=float(row['news_sentiment']),
                news_sentiment_5d=float(row['news_sentiment_5d']),
                dividend_yield=float(row['dividend_yield']),
                eps_surprise=float(row['eps_surprise'])
            ).on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_={c: getattr(Feature, c) for c in placeholders + [
                    'rsi_14', 'macd_signal', 'bb_pct', 'atr_14', 'obv_chg', 
                    'vol_ratio', 'stoch_k', 'ma_cross', 'high52w_ratio', 
                    'ret_1m', 'ret_3m'
                ]}
            )
            db_session.execute(stmt)
        
        db_session.commit()
        return df

    def process_all_tickers(self, tickers: list[str], db_session: Session, progress_callback=None):
        for ticker in tickers:
            try:
                self.build_features(ticker, db_session)
                if progress_callback:
                    progress_callback(ticker)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")
                db_session.rollback()
