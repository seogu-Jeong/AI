import sys
import os
import numpy as np
import pandas as pd
from datetime import date
from sqlalchemy import select, delete, desc
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Add project root to path to ensure imports work correctly
sys.path.append(os.getcwd())

from db.database import SessionLocal, init_db
from db.models import RawOHLCV, AIScore
from models.ensemble import EnsembleScorer
from data.feature_store import FeatureStore
from data.image_renderer import ImageRenderer

def run_inference():
    """
    Runs inference using trained models on all available tickers.
    Saves results to the AIScore table in the database.
    """
    print("Initializing Database...")
    init_db()
    db = SessionLocal()
    
    # Only delete TODAY's scores before re-generating — never wipe historical data
    today_date = date.today()
    print(f"Clearing today's AI scores ({today_date}) for fresh inference...")
    db.execute(delete(AIScore).where(AIScore.date == today_date))
    db.commit()
    
    print("Loading Ensemble Models and Feature Store...")
    try:
        ensemble = EnsembleScorer()
        renderer = ImageRenderer()
    except Exception as e:
        print(f"Failed to initialize models/renderer: {e}")
        return

    fs = FeatureStore()
    
    # Get all distinct tickers from RawOHLCV table
    stmt = select(RawOHLCV.ticker).distinct()
    tickers = db.execute(stmt).scalars().all()
    total = len(tickers)
    
    if total == 0:
        print("No tickers found in RawOHLCV table. Please seed data first.")
        db.close()
        return

    # Use latest OHLCV date so AIScore.date matches RawOHLCV.date for joins
    from db.models import RawOHLCV as OHLCV
    from sqlalchemy import func as sqfunc
    latest_ohlcv_date = db.execute(select(sqfunc.max(OHLCV.date))).scalar()
    today = latest_ohlcv_date or date.today()

    print(f"Found {total} tickers. Starting inference for {today}...")
    results = []
    
    for i, ticker in enumerate(tickers):
        try:
            # a. Get latest features for the ticker
            features = fs.get_latest_features(ticker, db)
            
            # b & c. Run real model inference or fall back to mock score
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
                        'Date': r.date,
                        'Open': r.open,
                        'High': r.high,
                        'Low': r.low,
                        'Close': r.close,
                        'Volume': r.volume
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
            
            # d. Collect data for scaling later
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
                'mlp_sell_prob': res['mlp_sell_prob'],
                'mode': mode
            }
            results.append(score_data)
            
        except Exception as e:
            print(f"[{i+1}/{total}] {ticker}: Error during inference - {e}")

    # --- Min-Max Rescaling to [20, 90] range ---
    if results:
        raw_scores = [r['ensemble_score'] for r in results]
        s_min, s_max = min(raw_scores), max(raw_scores)
        target_min, target_max = 20.0, 90.0
        
        print(f"\nRescaling scores from [{s_min:.2f}, {s_max:.2f}] to [{target_min}, {target_max}]...")
        
        for r in results:
            if s_max > s_min:
                # Scale ensemble score
                new_val = target_min + (r['ensemble_score'] - s_min) / (s_max - s_min) * (target_max - target_min)
                r['ensemble_score'] = max(0.0, min(100.0, new_val))
                
                # Also scale model sub-scores proportionally
                for key in ['lstm_score', 'cnn_score', 'transformer_score', 'mlp_score']:
                    if key in r:
                        new_sub = target_min + (r[key] - s_min) / (s_max - s_min) * (target_max - target_min)
                        r[key] = max(0.0, min(100.0, new_sub))
            else:
                r['ensemble_score'] = (target_min + target_max) / 2
                for key in ['lstm_score', 'cnn_score', 'transformer_score', 'mlp_score']:
                    if key in r:
                        r[key] = (target_min + target_max) / 2

        # Save rescaled results to DB
        success_count = 0
        for r in results:
            try:
                mode = r.pop('mode')
                stmt_insert = sqlite_insert(AIScore).values(**r).on_conflict_do_nothing(
                    index_elements=['ticker', 'date']
                )
                db.execute(stmt_insert)
                success_count += 1
                print(f"[{success_count}/{len(results)}] {r['ticker']}: {r['ensemble_score']:.2f} ({mode})")
            except Exception as e:
                print(f"Error saving {r['ticker']}: {e}")
                db.rollback()
        
        db.commit()
    else:
        success_count = 0

    # 7. Print summary at end
    print("\n" + "="*30)
    print("      INFERENCE SUMMARY")
    print("="*30)
    print(f"Date:               {today}")
    print(f"Total Tickers:      {total}")
    print(f"Successes:          {success_count}")
    print(f"Failures/Errors:    {total - success_count}")
    print("="*30)
    
    db.close()

if __name__ == "__main__":
    run_inference()
