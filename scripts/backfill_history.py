import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select, func as sqfunc, desc
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal, init_db
from db.models import RawOHLCV, Feature, AIScore
from data.collector import DataCollector
from data.preprocessor import Preprocessor
from data.image_renderer import ImageRenderer
from models.ensemble import EnsembleScorer

def get_features_for_date(ticker: str, target_date: date, db_session):
    """
    Helper to extract features for a specific date for inference.
    Similar to FeatureStore.get_latest_features but for any date.
    """
    # Get 80 OHLCV rows up to target_date to ensure we have 60 after rolling
    stmt_ohlcv = select(RawOHLCV).where(
        RawOHLCV.ticker == ticker, 
        RawOHLCV.date <= target_date
    ).order_by(desc(RawOHLCV.date)).limit(80)
    ohlcv_rows = db_session.execute(stmt_ohlcv).scalars().all()
    ohlcv_rows.reverse()
    
    # Get 1 Feature row for target_date
    stmt_feat = select(Feature).where(
        Feature.ticker == ticker, 
        Feature.date == target_date
    ).limit(1)
    feat_row = db_session.execute(stmt_feat).scalars().first()
    
    if not ohlcv_rows or not feat_row or len(ohlcv_rows) < 60:
        return {}
        
    df = pd.DataFrame([{
        'date': r.date,
        'open': r.open, 
        'high': r.high, 
        'low': r.low, 
        'close': r.close, 
        'volume': r.volume
    } for r in ohlcv_rows])
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    
    # Get 60 Feature rows up to target_date for LSTM sequence indicators
    stmt_feat_60 = select(Feature).where(
        Feature.ticker == ticker, 
        Feature.date <= target_date
    ).order_by(desc(Feature.date)).limit(60)
    feat_rows_60 = db_session.execute(stmt_feat_60).scalars().all()
    feat_rows_60.reverse()
    
    if len(feat_rows_60) < 60:
        return {}
    
    feat_df_60 = pd.DataFrame([{
        'date': r.date,
        'rsi_14': r.rsi_14, 
        'macd_signal': r.macd_signal, 
        'atr_14': r.atr_14
    } for r in feat_rows_60])
    
    # Align by date
    merged = pd.merge(df, feat_df_60, on='date')
    if len(merged) < 60:
        return {}
    
    lstm_data = merged.tail(60).reset_index(drop=True)
    lstm_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'rsi_14', 'macd_signal', 'atr_14']
    
    trans_cols = [
        'rsi_14', 'macd_signal', 'bb_pct', 'atr_14', 'obv_chg', 'vol_ratio', 'stoch_k', 'ma_cross',
        'per', 'pbr', 'roe', 'eps_growth', 'debt_ratio', 'op_margin', 'market_cap', 'high52w_ratio',
        'ret_1m', 'ret_3m', 'usdkrw_chg', 'vix', 'news_sentiment', 'news_sentiment_5d', 
        'dividend_yield', 'eps_surprise'
    ]
    trans_vec = [getattr(feat_row, c) for c in trans_cols]
    
    lstm_seq = lstm_data[lstm_cols].values
    s_min = lstm_seq.min(axis=0)
    s_max = lstm_seq.max(axis=0)
    norm_lstm_seq = (lstm_seq - s_min) / (s_max - s_min + 1e-8)
    
    return {
        'lstm_seq': norm_lstm_seq.reshape(1, 60, 10),
        'transformer_vec': np.array(trans_vec).reshape(1, 24)
    }

def main():
    print("Starting History Backfill Script...")
    init_db()
    db = SessionLocal()
    
    collector = DataCollector()
    preprocessor = Preprocessor()
    tickers = collector.get_sp500_tickers()
    
    # Step 1: Download OHLCV from 2019-01-01 to today
    print(f"\n[Step 1/3] Downloading OHLCV for {len(tickers)} tickers...")
    start_date = "2016-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    def progress_ohlcv(t):
        print(f"  Downloaded: {t}")
        
    collector.collect_ohlcv(tickers, start_date, end_date, db, progress_callback=progress_ohlcv)
    
    # Step 2: Rebuild all Feature rows
    print("\n[Step 2/3] Rebuilding Feature rows...")
    for i, ticker in enumerate(tickers):
        try:
            print(f"  [{i+1}/{len(tickers)}] Processing features for {ticker}...")
            preprocessor.build_features(ticker, db)
        except Exception as e:
            print(f"  Error rebuilding features for {ticker}: {e}")
            db.rollback()
            
    # Step 3: Backfill AIScore monthly
    print("\n[Step 3/3] Backfilling AIScore for monthly rebalance dates...")
    try:
        ensemble = EnsembleScorer()
        renderer = ImageRenderer()
    except Exception as e:
        print(f"Failed to initialize models/renderer: {e}")
        return

    rebalance_months = pd.date_range(start='2019-07-01', end=datetime.now(), freq='MS')
    total_dates_processed = 0
    total_scores_saved = 0

    for month_start in rebalance_months:
        # Find first available trading date for this month
        target_date = db.execute(select(sqfunc.min(RawOHLCV.date)).where(
            RawOHLCV.date >= month_start.date(),
            RawOHLCV.date < (month_start + pd.offsets.MonthBegin(1)).date()
        )).scalar()
        
        if not target_date:
            continue
            
        print(f"\nProcessing date: {target_date}")
        results = []
        
        for ticker in tickers:
            try:
                features = get_features_for_date(ticker, target_date, db)
                
                if features and 'lstm_seq' in features and 'transformer_vec' in features:
                    # Fetch last 20 OHLCV rows for image rendering
                    stmt_ohlcv_20 = select(RawOHLCV).where(
                        RawOHLCV.ticker == ticker,
                        RawOHLCV.date <= target_date
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
                        else:
                            cnn_image = arr.transpose(2, 0, 1).astype(np.float32)[np.newaxis]
                            res = ensemble.compute_score(
                                ticker,
                                features['lstm_seq'],
                                cnn_image,
                                features['transformer_vec']
                            )
                    else:
                        res = ensemble.generate_mock_score(ticker)
                else:
                    res = ensemble.generate_mock_score(ticker)
                
                score_data = {
                    'ticker': ticker,
                    'date': target_date,
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
                results.append(score_data)
            except Exception as e:
                print(f"  Error processing {ticker} for {target_date}: {e}")

        # Min-Max Rescaling for the current date's batch
        if results:
            raw_scores = [r['ensemble_score'] for r in results]
            s_min, s_max = min(raw_scores), max(raw_scores)
            target_min, target_max = 20.0, 90.0
            
            for r in results:
                if s_max > s_min:
                    new_val = target_min + (r['ensemble_score'] - s_min) / (s_max - s_min) * (target_max - target_min)
                    r['ensemble_score'] = max(0.0, min(100.0, new_val))
                    
                    # Scale model sub-scores proportionally
                    for key in ['lstm_score', 'cnn_score', 'transformer_score', 'mlp_score']:
                        new_sub = target_min + (r[key] - s_min) / (s_max - s_min) * (target_max - target_min)
                        r[key] = max(0.0, min(100.0, new_sub))
                else:
                    r['ensemble_score'] = (target_min + target_max) / 2
                    for key in ['lstm_score', 'cnn_score', 'transformer_score', 'mlp_score']:
                        r[key] = (target_min + target_max) / 2
            
            # Save batch to DB
            for r in results:
                try:
                    stmt = sqlite_insert(AIScore).values(**r).on_conflict_do_nothing(
                        index_elements=['ticker', 'date']
                    )
                    db.execute(stmt)
                    total_scores_saved += 1
                except Exception as e:
                    print(f"  Error saving {r['ticker']} for {target_date}: {e}")
            
            db.commit()
            print(f"  Processed {len(results)} tickers for {target_date}")
            total_dates_processed += 1

    # Summary
    print("\n" + "="*40)
    print("        BACKFILL COMPLETE")
    print("="*40)
    print(f"Total dates processed:    {total_dates_processed}")
    print(f"Total AIScore rows saved: {total_scores_saved}")
    
    db_path = "stocksense.db"
    if os.path.exists(db_path):
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"Database size:            {db_size_mb:.2f} MB")
    
    print("="*40)
    db.close()

if __name__ == "__main__":
    main()
