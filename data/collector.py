import pandas as pd
import yfinance as yf
import httpx
import io
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from db.models import RawOHLCV
from db.database import SessionLocal
from sqlalchemy.orm import Session

class DataCollector:
    SP500_TICKERS_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    @staticmethod
    def get_sp500_tickers() -> list[str]:
        try:
            import ssl, urllib.request
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            req = urllib.request.Request(DataCollector.SP500_TICKERS_URL, headers=headers)
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
            html = resp.read().decode('utf-8')
            tables = pd.read_html(io.StringIO(html), flavor='lxml')
            df = tables[0]
            tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
            print(f"Fetched {len(tickers)} S&P500 tickers from Wikipedia")
            return tickers
        except Exception as e:
            print(f"Error scraping S&P 500 tickers: {e}")
            return DataCollector.get_cached_sp500_tickers()

    @staticmethod
    def get_sp500_tickers_with_sectors() -> dict[str, str]:
        """Returns a mapping of ticker -> GICS Sector"""
        try:
            import ssl, urllib.request
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
            req = urllib.request.Request(DataCollector.SP500_TICKERS_URL, headers=headers)
            resp = urllib.request.urlopen(req, context=ctx, timeout=15)
            html = resp.read().decode('utf-8')
            tables = pd.read_html(io.StringIO(html), flavor='lxml')
            df = tables[0]
            ticker_sector = dict(zip([t.replace('.', '-') for t in df['Symbol']], df['GICS Sector']))
            return ticker_sector
        except Exception as e:
            print(f"Error scraping S&P 500 sectors: {e}")
            return {}

    @staticmethod
    def get_cached_sp500_tickers() -> list[str]:
        return ['AAPL','MSFT','NVDA','AMZN','GOOGL','META','BRK-B','LLY','AVGO','JPM','UNH','XOM','TSLA','PG','MA','JNJ','COST','HD','MRK','ABBV','CVX','BAC','NFLX','AMD','CRM','PEP','TMO','ADBE','ACN','LIN','MCD','DIS','WMT','TXN','VZ','INTC','PM','NEE','RTX','QCOM','HON','AMGN','IBM','INTU','CAT','SPGI','GS','BLK','MS','SCHW']

    def collect_ohlcv(self, tickers: list[str], start: str, end: str, db_session: Session, progress_callback=None):
        def download_and_store(ticker):
            local_session = SessionLocal()
            try:
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                if df.empty:
                    return
                # Flatten MultiIndex columns (yfinance 0.2+ returns ('Close', 'AAPL') etc.)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                rows_to_insert = []
                for dt, row in df.iterrows():
                    rows_to_insert.append({
                        'ticker': ticker,
                        'date': dt.date(),
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']),
                        'adj_close': float(row['Close']),
                        'dividend': 0.0,
                    })

                if rows_to_insert:
                    local_session.execute(
                        sqlite_insert(RawOHLCV).on_conflict_do_nothing(),
                        rows_to_insert
                    )
                    local_session.commit()

                if progress_callback:
                    progress_callback(ticker)
            except Exception as e:
                print(f"Error collecting {ticker}: {e}")
                local_session.rollback()
            finally:
                local_session.close()

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(download_and_store, tickers)

    def collect_fred_rates(self, db_session: Session):
        """
        Collects USD/KRW exchange rates via yfinance fallback (skipping FRED API).
        Stores them as 'USD_KRW' in the RawOHLCV table.
        """
        print("Collecting USD/KRW exchange rates via yfinance...")
        local_session = SessionLocal()
        try:
            # Download USDKRW=X for the last 30 days
            ticker = "USDKRW=X"
            df = yf.download(ticker, period="30d", progress=False, auto_adjust=True)
            
            if df.empty:
                print("No data found for USD/KRW.")
                return

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            rows_to_insert = []
            for dt, row in df.iterrows():
                rows_to_insert.append({
                    'ticker': 'USD_KRW',
                    'date': dt.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row.get('Volume', 0)),
                    'adj_close': float(row['Close']),
                    'dividend': 0.0,
                })

            if rows_to_insert:
                local_session.execute(
                    sqlite_insert(RawOHLCV).on_conflict_do_nothing(),
                    rows_to_insert
                )
                local_session.commit()
                print(f"Stored {len(rows_to_insert)} days of USD/KRW data.")
                
        except Exception as e:
            print(f"Error collecting USD/KRW: {e}")
            local_session.rollback()
        finally:
            local_session.close()
