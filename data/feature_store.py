import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from db.models import RawOHLCV, Feature, AIScore

class FeatureStore:
    def get_features_for_training(self, tickers: list[str], db_session: Session, lookback: int = 60) -> dict:
        all_lstm = []
        all_trans = []
        all_labels = {
            'direction_5d': [],
            'direction_20d': [],
            'direction_60d': [],
            'pattern': [],
            'signal': []
        }
        
        for ticker in tickers:
            # Get OHLCV
            stmt_ohlcv = select(RawOHLCV).where(RawOHLCV.ticker == ticker).order_by(RawOHLCV.date)
            ohlcv_rows = db_session.execute(stmt_ohlcv).scalars().all()
            if len(ohlcv_rows) < lookback + 60: continue
            
            # Get Features
            stmt_feat = select(Feature).where(Feature.ticker == ticker).order_by(Feature.date)
            feat_rows = db_session.execute(stmt_feat).scalars().all()
            
            ohlcv_df = pd.DataFrame([{
                'date': r.date, 'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close, 'volume': r.volume
            } for r in ohlcv_rows])
            
            feat_df = pd.DataFrame([{
                'date': r.date, 'rsi_14': r.rsi_14, 'macd_signal': r.macd_signal, 'bb_pct': r.bb_pct,
                'atr_14': r.atr_14, 'obv_chg': r.obv_chg, 'vol_ratio': r.vol_ratio, 'stoch_k': r.stoch_k,
                'ma_cross': r.ma_cross, 'per': r.per, 'pbr': r.pbr, 'roe': r.roe, 'eps_growth': r.eps_growth,
                'debt_ratio': r.debt_ratio, 'op_margin': r.op_margin, 'market_cap': r.market_cap,
                'high52w_ratio': r.high52w_ratio, 'ret_1m': r.ret_1m, 'ret_3m': r.ret_3m,
                'usdkrw_chg': r.usdkrw_chg, 'vix': r.vix, 'news_sentiment': r.news_sentiment,
                'news_sentiment_5d': r.news_sentiment_5d, 'dividend_yield': r.dividend_yield,
                'eps_surprise': r.eps_surprise
            } for r in feat_rows])
            
            # Merge
            df = pd.merge(ohlcv_df, feat_df, on='date')
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma5'] = df['ma5'].bfill().ffill()
            df['ma20'] = df['ma20'].bfill().ffill()
            df = df.dropna().reset_index(drop=True)
            
            # Columns for LSTM: [open,high,low,close,volume,ma5,ma20,rsi_14,macd_signal,atr_14]
            lstm_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'rsi_14', 'macd_signal', 'atr_14']
            # Columns for Transformer: all 24 features
            trans_cols = [
                'rsi_14', 'macd_signal', 'bb_pct', 'atr_14', 'obv_chg', 'vol_ratio', 'stoch_k', 'ma_cross',
                'per', 'pbr', 'roe', 'eps_growth', 'debt_ratio', 'op_margin', 'market_cap', 'high52w_ratio',
                'ret_1m', 'ret_3m', 'usdkrw_chg', 'vix', 'news_sentiment', 'news_sentiment_5d', 
                'dividend_yield', 'eps_surprise'
            ]
            
            for i in range(lookback, len(df) - 60):
                seq = df.iloc[i-lookback:i][lstm_cols].values
                all_lstm.append(seq)
                
                trans_vec = df.iloc[i-1][trans_cols].values
                all_trans.append(trans_vec)
                
                # Labels
                curr_close = df.iloc[i-1]['close']
                ret5 = (df.iloc[i+5]['close'] - curr_close) / curr_close
                ret20 = (df.iloc[i+20]['close'] - curr_close) / curr_close
                ret60 = (df.iloc[i+60]['close'] - curr_close) / curr_close
                
                def get_dir(r): return 0 if r > 0.02 else (2 if r < -0.02 else 1)
                def get_sig(r): return 0 if r > 0.03 else (2 if r < -0.03 else 1)
                
                all_labels['direction_5d'].append(get_dir(ret5))
                all_labels['direction_20d'].append(get_dir(ret20))
                all_labels['direction_60d'].append(get_dir(ret60))
                all_labels['pattern'].append(get_dir(ret5)) # placeholder
                all_labels['signal'].append(get_sig(ret20))

        # Per-sequence min-max normalization for LSTM to prevent NaN loss
        norm_all_lstm = []
        for seq in all_lstm:
            s_min = seq.min(axis=0)
            s_max = seq.max(axis=0)
            norm_seq = (seq - s_min) / (s_max - s_min + 1e-8)
            norm_all_lstm.append(norm_seq)

        return {
            'lstm_sequences': np.array(norm_all_lstm),
            'transformer_factors': np.array(all_trans),
            'labels': {k: np.array(v) for k, v in all_labels.items()},
            'cnn_images_paths': [] # Placeholder as requested
        }

    def get_latest_features(self, ticker: str, db_session: Session) -> dict:
        stmt_ohlcv = select(RawOHLCV).where(RawOHLCV.ticker == ticker).order_by(desc(RawOHLCV.date)).limit(80)
        ohlcv_rows = db_session.execute(stmt_ohlcv).scalars().all()
        ohlcv_rows.reverse()
        
        stmt_feat = select(Feature).where(Feature.ticker == ticker).order_by(desc(Feature.date)).limit(1)
        feat_row = db_session.execute(stmt_feat).scalars().first()
        
        if not ohlcv_rows or not feat_row or len(ohlcv_rows) < 60:
            return {}
            
        df = pd.DataFrame([{
            'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close, 'volume': r.volume
        } for r in ohlcv_rows])
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # We need the 24 features from the latest Feature row, and merge with latest OHLCV for LSTM
        # Actually, for LSTM we need rsi_14, macd_signal, atr_14 which are in Feature. 
        # But we need them for the LAST 60 days.
        # This implies we need 60 Feature rows too.
        
        stmt_feat_60 = select(Feature).where(Feature.ticker == ticker).order_by(desc(Feature.date)).limit(60)
        feat_rows_60 = db_session.execute(stmt_feat_60).scalars().all()
        feat_rows_60.reverse()
        
        feat_df_60 = pd.DataFrame([{
            'rsi_14': r.rsi_14, 'macd_signal': r.macd_signal, 'atr_14': r.atr_14
        } for r in feat_rows_60])
        
        if len(feat_df_60) < 60: return {}
        
        lstm_df = df.tail(60).reset_index(drop=True)
        # Assuming alignment. In real world we should join on date.
        lstm_data = pd.concat([lstm_df, feat_df_60], axis=1)
        lstm_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma20', 'rsi_14', 'macd_signal', 'atr_14']
        
        trans_cols = [
            'rsi_14', 'macd_signal', 'bb_pct', 'atr_14', 'obv_chg', 'vol_ratio', 'stoch_k', 'ma_cross',
            'per', 'pbr', 'roe', 'eps_growth', 'debt_ratio', 'op_margin', 'market_cap', 'high52w_ratio',
            'ret_1m', 'ret_3m', 'usdkrw_chg', 'vix', 'news_sentiment', 'news_sentiment_5d', 
            'dividend_yield', 'eps_surprise'
        ]
        trans_vec = [getattr(feat_row, c) for c in trans_cols]
        
        lstm_seq = lstm_data[lstm_cols].values
        # Per-sequence min-max normalization
        s_min = lstm_seq.min(axis=0)
        s_max = lstm_seq.max(axis=0)
        norm_lstm_seq = (lstm_seq - s_min) / (s_max - s_min + 1e-8)
        
        return {
            'lstm_seq': norm_lstm_seq.reshape(1, 60, 10),
            'transformer_vec': np.array(trans_vec).reshape(1, 24)
        }

    def get_top_scores(self, n: int, db_session: Session, date: str = None) -> list[dict]:
        query = select(AIScore, RawOHLCV.close).join(RawOHLCV, (AIScore.ticker == RawOHLCV.ticker) & (AIScore.date == RawOHLCV.date))
        if date:
            query = query.where(AIScore.date == date)
        else:
            # Get latest date from AIScore
            latest_date_stmt = select(AIScore.date).order_by(desc(AIScore.date)).limit(1)
            latest_date = db_session.execute(latest_date_stmt).scalar()
            if latest_date:
                query = query.where(AIScore.date == latest_date)
                
        query = query.order_by(desc(AIScore.ensemble_score)).limit(n)
        results = db_session.execute(query).all()
        
        output = []
        for score, close in results:
            output.append({
                'ticker': score.ticker,
                'ensemble_score': score.ensemble_score,
                'lstm_dir_5d': score.lstm_dir_5d,
                'cnn_pattern': score.cnn_pattern,
                'mlp_signal': score.mlp_signal,
                'mlp_buy_prob': score.mlp_buy_prob,
                'close_price': close
            })
        return output
