import os
import sys
from datetime import datetime, timedelta
from db.database import init_db, SessionLocal
from data.collector import DataCollector
from data.preprocessor import Preprocessor

def seed_data():
    print("Initializing Database...")
    init_db()
    
    db = SessionLocal()
    collector = DataCollector()
    preprocessor = Preprocessor()
    
    print("Fetching tickers...")
    tickers = collector.get_cached_sp500_tickers()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    print(f"Collecting OHLCV data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
    
    def on_collect(ticker):
        print(f" Collected {ticker}")
        
    collector.collect_ohlcv(
        tickers=tickers, 
        start=start_date.strftime("%Y-%m-%d"), 
        end=end_date.strftime("%Y-%m-%d"), 
        db_session=db,
        progress_callback=on_collect
    )
    
    print("Running Preprocessor...")
    
    def on_process(ticker):
        print(f" Processed {ticker}")
        
    preprocessor.process_all_tickers(
        tickers=tickers,
        db_session=db,
        progress_callback=on_process
    )
    
    db.close()
    print("Seed data complete.")

if __name__ == "__main__":
    seed_data()
