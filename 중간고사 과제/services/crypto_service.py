import numpy as np
import pandas as pd
import yfinance as yf

COINS = [
    {'ticker': 'BTC-USD',  'name': 'Bitcoin',   'symbol': 'BTC'},
    {'ticker': 'ETH-USD',  'name': 'Ethereum',  'symbol': 'ETH'},
    {'ticker': 'BNB-USD',  'name': 'BNB',       'symbol': 'BNB'},
    {'ticker': 'SOL-USD',  'name': 'Solana',    'symbol': 'SOL'},
    {'ticker': 'XRP-USD',  'name': 'Ripple',    'symbol': 'XRP'},
    {'ticker': 'ADA-USD',  'name': 'Cardano',   'symbol': 'ADA'},
    {'ticker': 'DOGE-USD', 'name': 'Dogecoin',  'symbol': 'DOGE'},
    {'ticker': 'AVAX-USD', 'name': 'Avalanche', 'symbol': 'AVAX'},
    {'ticker': 'DOT-USD',  'name': 'Polkadot',  'symbol': 'DOT'},
    {'ticker': 'LINK-USD', 'name': 'Chainlink', 'symbol': 'LINK'},
]


class CryptoService:
    @staticmethod
    def get_all_scores() -> list:
        results = []
        for coin in COINS:
            try:
                d = CryptoService.analyze(coin['ticker'], coin['name'], coin['symbol'])
                if d:
                    results.append(d)
            except Exception as e:
                print(f"Crypto error {coin['ticker']}: {e}")
        return sorted(results, key=lambda x: x['score'], reverse=True)

    @staticmethod
    def analyze(ticker: str, name: str, symbol: str) -> dict:
        df = yf.download(ticker, period='6mo', interval='1d', progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        close = df['Close'].squeeze()
        high  = df['High'].squeeze()
        low   = df['Low'].squeeze()

        # RSI(14)
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_series = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        rsi = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_val = float((ema12 - ema26).iloc[-1])

        # Bollinger Band %B
        sma20  = close.rolling(20).mean()
        std20  = close.rolling(20).std()
        bb_top = sma20 + 2 * std20
        bb_bot = sma20 - 2 * std20
        bb_range = (bb_top - bb_bot).iloc[-1]
        bb_pct = float((close.iloc[-1] - bb_bot.iloc[-1]) / bb_range) if bb_range > 0 else 0.5

        # Stoch %K
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        h_l = (high14 - low14).iloc[-1]
        stoch = float((close.iloc[-1] - low14.iloc[-1]) / h_l * 100) if h_l > 0 else 50.0

        # 1-month momentum
        mom_1m = float((close.iloc[-1] / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0

        # 7-day momentum
        mom_7d = float((close.iloc[-1] / close.iloc[-8] - 1) * 100) if len(close) >= 8 else 0.0

        # --- Composite score (0-100) ---
        score = 50.0

        # RSI contribution
        if rsi < 30:    score += 15
        elif rsi < 45:  score += 7
        elif rsi > 70:  score -= 15
        elif rsi > 58:  score -= 7

        # MACD contribution (normalize by price)
        price_now = float(close.iloc[-1])
        if price_now > 0:
            macd_norm = macd_val / price_now * 1000
            score += float(np.clip(macd_norm * 3, -12, 12))

        # BB%B contribution
        if bb_pct < 0.2:    score += 10
        elif bb_pct < 0.35: score += 5
        elif bb_pct > 0.8:  score -= 10
        elif bb_pct > 0.65: score -= 5

        # Stoch contribution
        if stoch < 20:    score += 8
        elif stoch < 35:  score += 4
        elif stoch > 80:  score -= 8
        elif stoch > 65:  score -= 4

        # Momentum contribution
        score += float(np.clip(mom_7d * 0.8, -12, 12))

        score = float(np.clip(score, 0, 100))

        if score >= 70:   signal = 'BUY'
        elif score >= 50: signal = 'HOLD'
        else:             signal = 'SELL'

        change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0.0

        # 24h high/low
        high_24h = float(high.iloc[-1])
        low_24h  = float(low.iloc[-1])

        # Market cap rank proxy (volume * price)
        vol = df['Volume'].squeeze()
        volume_24h = float(vol.iloc[-1]) if not vol.empty else 0.0

        return {
            'ticker':     ticker,
            'name':       name,
            'symbol':     symbol,
            'price':      price_now,
            'change_pct': change_pct,
            'score':      score,
            'signal':     signal,
            'rsi':        rsi,
            'macd':       macd_val,
            'bb_pct':     bb_pct,
            'stoch':      stoch,
            'mom_1m':     mom_1m,
            'mom_7d':     mom_7d,
            'high_24h':   high_24h,
            'low_24h':    low_24h,
            'volume_24h': volume_24h,
            'history':    df,
        }
