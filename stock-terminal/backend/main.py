from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYMBOLS = [
    "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "XOM", "PG", "HD", "CVX", "MRK", "ABBV", "KO", "PEP",
    "BAC", "AVGO", "COST", "MCD", "WMT", "TMO", "CSCO", "ABT", "CRM", "DIS",
    "PFE", "NFLX", "LIN", "AMD", "NKE", "CMCSA", "ADBE", "TXN", "PM", "VZ",
    "INTC", "QCOM", "HON", "IBM", "AMGN", "NOW", "GE", "BA", "ISRG", "CAT"
]

def generate_stock_data(symbol):
    price = round(random.uniform(50, 1000), 2)
    change = round(random.uniform(-10, 10), 2)
    change_pct = round((change / price) * 100, 2)
    ai_score = random.randint(30, 95)
    
    if ai_score > 70:
        signal = 'BUY'
    elif ai_score < 40:
        signal = 'SELL'
    else:
        signal = 'HOLD'
        
    return {
        "symbol": symbol,
        "name": f"{symbol} Corporation",
        "price": price,
        "change": change,
        "change_pct": change_pct,
        "ai_score": ai_score,
        "signal": signal,
        "volume": f"{random.randint(10, 100)}.{random.randint(1, 9)}M",
        "market_cap": f"{random.randint(1, 3)}.{random.randint(1, 9)}T" if random.random() > 0.5 else f"{random.randint(100, 900)}B",
        "sector": random.choice(["Technology", "Healthcare", "Finance", "Consumer", "Energy"]),
        "lstm_score": random.randint(40, 95),
        "cnn_score": random.randint(40, 95),
        "transformer_score": random.randint(40, 95),
        "sentiment_score": random.randint(40, 95)
    }

STOCKS = [generate_stock_data(sym) for sym in SYMBOLS]

@app.get("/api/stocks")
def get_stocks():
    return STOCKS

@app.get("/api/stock/{symbol}")
def get_stock(symbol: str):
    for s in STOCKS:
        if s["symbol"] == symbol.upper():
            return s
    return {"error": "Not found"}

@app.get("/api/news/{symbol}")
def get_news(symbol: str):
    return [
        {"id": 1, "headline": f"{symbol} announces new breakthrough product", "source": "Reuters", "time": "10:30 AM", "sentiment": "Positive"},
        {"id": 2, "headline": f"Analysts update {symbol} price target", "source": "Bloomberg", "time": "09:15 AM", "sentiment": "Neutral"},
        {"id": 3, "headline": f"Supply chain issues affect {symbol}", "source": "WSJ", "time": "Yesterday", "sentiment": "Negative"},
    ]

@app.get("/api/market-overview")
def get_market_overview():
    return {
        "SP500": {"value": 5234.18, "change": 45.2, "change_pct": 0.8},
        "NASDAQ": {"value": 16340.5, "change": 120.4, "change_pct": 0.74},
        "DOW": {"value": 39400.2, "change": -13.4, "change_pct": -0.03},
        "VIX": {"value": 14.2, "change": -0.2, "change_pct": -1.5}
    }
