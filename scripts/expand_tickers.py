import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import select, desc, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Add project root to path to ensure imports work correctly
sys.path.append(os.getcwd())

from db.database import SessionLocal, init_db
from db.models import RawOHLCV, AIScore
from data.collector import DataCollector
from data.preprocessor import Preprocessor
from data.feature_store import FeatureStore
from data.image_renderer import ImageRenderer
from models.ensemble import EnsembleScorer

def main():
    print("Starting Ticker Expansion Script...")
    init_db()
    db = SessionLocal()
    
    collector = DataCollector()
    preprocessor = Preprocessor()
    fs = FeatureStore()
    renderer = ImageRenderer()
    
    # Load Ensemble Models
    try:
        ensemble = EnsembleScorer()
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return
    
    # a. Get full S&P500 list via DataCollector.get_sp500_tickers()
    print("Fetching full S&P 500 list from Wikipedia...")
    full_list = collector.get_sp500_tickers()
    
    # b. Get tickers already in RawOHLCV DB
    stmt = select(RawOHLCV.ticker).distinct()
    existing = db.execute(stmt).scalars().all()
    existing_set = set(existing)
    
    # c. new_tickers = full_list minus existing_tickers
    new_tickers = [t for t in full_list if t not in existing_set]
    
    # d. Print status
    print(f"Found {len(full_list)} S&P500 tickers, {len(existing)} already in DB, {len(new_tickers)} new")
    
    # e. If no new tickers, exit
    if not new_tickers:
        print("All tickers already in DB.")
        db.close()
        return

    # f. Download last 2 years OHLCV for new_tickers
    print(f"\n[Step 1/3] Downloading 2 years of OHLCV for {len(new_tickers)} new tickers...")
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365*2)).strftime("%Y-%m-%d")
    
    def progress_ohlcv(t):
        print(f"  Downloaded: {t}")
        
    collector.collect_ohlcv(new_tickers, start_date, end_date, db, progress_callback=progress_ohlcv)
    
    # g. Build features for new_tickers
    print(f"\n[Step 2/3] Building features for {len(new_tickers)} tickers...")
    for i, ticker in enumerate(new_tickers):
        try:
            preprocessor.build_features(ticker, db)
            print(f"  [{i+1}/{len(new_tickers)}] Features built: {ticker}")
        except Exception as e:
            print(f"  [{i+1}/{len(new_tickers)}] Error building features for {ticker}: {e}")
            db.rollback()

    # h. Run inference for new tickers for today's date
    # Use latest OHLCV date so AIScore.date matches RawOHLCV.date
    latest_ohlcv_date = db.execute(select(func.max(RawOHLCV.date))).scalar()
    today = latest_ohlcv_date or date.today()
    
    print(f"\n[Step 3/3] Running inference for {len(new_tickers)} tickers for {today}...")
    
    results_count = 0
    for i, ticker in enumerate(new_tickers):
        try:
            # a. Get latest features
            features = fs.get_latest_features(ticker, db)
            
            # b & c. Run inference
            if features and 'lstm_seq' in features and 'transformer_vec' in features:
                 # Fetch last 20 OHLCV rows for image rendering
                stmt_ohlcv_20 = select(RawOHLCV).where(
                    RawOHLCV.ticker == ticker,
                    RawOHLCV.date <= today
                ).order_by(desc(RawOHLCV.date)).limit(20)
                rows_20 = db.execute(stmt_ohlcv_20).scalars().all()
                
                if len(rows_20) == 20:
                    rows_20.reverse()
                    df_img = pd.DataFrame([{
                        'Date': r.date, 'Open': r.open, 'High': r.high, 'Low': r.low, 'Close': r.close, 'Volume': r.volume
                    } for r in rows_20])
                    df_img.set_index('Date', inplace=True)
                    df_img.index = pd.to_datetime(df_img.index)
                    
                    arr = renderer.render_candle_image(df_img, ticker)
                    
                    if np.all(arr == 0):
                        res = ensemble.generate_mock_score(ticker)
                        mode = "MOCK_IMG_FAIL"
                    else:
                        cnn_image = arr.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
                        res = ensemble.compute_score(
                            ticker,
                            features['lstm_seq'],
                            cnn_image,
                            features['transformer_vec']
                        )
                        mode = "REAL"
                else:
                    res = ensemble.generate_mock_score(ticker)
                    mode = "MOCK_INSUFFICIENT_DATA"
            else:
                res = ensemble.generate_mock_score(ticker)
                mode = "MOCK_NO_FEATURES"
            
            score_data = {
                'ticker': res['ticker'],
                'date': today,
                'lstm_score': res['lstm_score'],
                'cnn_score': res['cnn_score'],
                'transformer_score': res['transformer_score'],
                'mlp_score': res['mlp_score'],
                'ensemble_score': res['ensemble_score'],
                'lstm_dir_5d': res['lstm_dir_5d'],
                'lstm_dir_20d': res.get('lstm_dir_20d'),
                'lstm_dir_60d': res.get('lstm_dir_60d'),
                'cnn_pattern': res['cnn_pattern'],
                'mlp_signal': res['mlp_signal'],
                'mlp_buy_prob': res['mlp_buy_prob'],
                'mlp_hold_prob': res['mlp_hold_prob'],
                'mlp_sell_prob': res['mlp_sell_prob']
            }
            
            stmt_insert = sqlite_insert(AIScore).values(**score_data).on_conflict_do_update(
                index_elements=['ticker', 'date'],
                set_=score_data
            )
            db.execute(stmt_insert)
            results_count += 1
            print(f"  [{i+1}/{len(new_tickers)}] Inference: {ticker} ({mode})")
            
        except Exception as e:
            print(f"  [{i+1}/{len(new_tickers)}] Error running inference for {ticker}: {e}")
            db.rollback()

    db.commit()
    
    # i. Print summary
    print("\n" + "="*30)
    print("      EXPANSION SUMMARY")
    print("="*30)
    print(f"New Tickers Added:  {len(new_tickers)}")
    print(f"AIScores Created:   {results_count}")
    print("="*30)
    
    db.close()

if __name__ == "__main__":
    main()
