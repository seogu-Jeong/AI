import datetime
import numpy as np
from PySide6.QtCore import QThread, Signal
from models.ensemble import EnsembleScorer
from db.database import SessionLocal
from db.models import AIScore, Feature
from sqlalchemy import select
import os
from PIL import Image

class ModelWorker(QThread):
    progress = Signal(int)
    score_ready = Signal(dict)
    all_done = Signal(list)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, tickers: list[str], use_mock: bool = False):
        super().__init__()
        self.tickers = tickers
        self.use_mock = use_mock
        self.scorer = None

    def run(self):
        db_session = SessionLocal()
        all_results = []
        try:
            self.status.emit('모델 로드 중...')
            self.scorer = EnsembleScorer()
            
            for i, ticker in enumerate(self.tickers):
                if self.isInterruptionRequested(): return
                
                self.status.emit(f'{ticker} 분석 중...')
                
                if self.use_mock:
                    result = self.scorer.generate_mock_score(ticker)
                else:
                    try:
                        # Fetch latest features for ticker
                        stmt = select(Feature).where(Feature.ticker == ticker).order_by(Feature.date.desc()).limit(60)
                        rows = db_session.execute(stmt).scalars().all()
                        
                        if not rows or len(rows) < 20:
                            # Not enough data, fallback to mock or skip
                            result = self.scorer.generate_mock_score(ticker)
                        else:
                            # Prepare inputs (simplified for this worker)
                            # In a real app, we'd need to properly format these
                            # For now, we'll try to get what we can or fallback to mock
                            
                            # LSTM seq (last 20 days of some features)
                            # Transformer factors (latest 24-30 features)
                            # CNN image from cache
                            from app_paths import get_user_file
                            img_path = str(get_user_file(f'cache/images/{ticker}.png'))
                            if os.path.exists(img_path):
                                img = Image.open(img_path).convert('RGB')
                                img = img.resize((128, 128))
                                cnn_image = np.array(img)
                            else:
                                cnn_image = np.zeros((128, 128, 3), dtype=np.uint8)
                            
                            # Factors from the latest row
                            latest_row = rows[0]
                            factors = np.array([
                                latest_row.rsi_14, latest_row.macd_signal, latest_row.bb_pct,
                                latest_row.atr_14, latest_row.obv_chg, latest_row.vol_ratio,
                                latest_row.stoch_k, latest_row.ma_cross, latest_row.per,
                                latest_row.pbr, latest_row.roe, latest_row.eps_growth,
                                latest_row.debt_ratio, latest_row.op_margin, latest_row.market_cap,
                                latest_row.high52w_ratio, latest_row.ret_1m, latest_row.ret_3m,
                                latest_row.usdkrw_chg, latest_row.vix, latest_row.news_sentiment,
                                latest_row.news_sentiment_5d, latest_row.dividend_yield, latest_row.eps_surprise
                            ])
                            
                            # LSTM dummy seq (needs 20x features, but we'll use latest row repeated for demo if needed)
                            # Real app would use actual history
                            lstm_seq = np.tile(factors, (20, 1)) 
                            
                            result = self.scorer.compute_score(ticker, lstm_seq, cnn_image, factors)
                    except Exception as e:
                        print(f"Inference error for {ticker}, falling back to mock: {e}")
                        result = self.scorer.generate_mock_score(ticker)
                
                # Save to DB
                ai_score = AIScore(
                    ticker=ticker,
                    date=datetime.datetime.now().date(),
                    lstm_score=result['lstm_score'],
                    cnn_score=result['cnn_score'],
                    transformer_score=result['transformer_score'],
                    mlp_score=result['mlp_score'],
                    ensemble_score=result['ensemble_score'],
                    lstm_dir_5d=result.get('lstm_dir_5d', 'N/A'),
                    cnn_pattern=result.get('cnn_pattern', 'N/A'),
                    mlp_signal=result.get('mlp_signal', 'N/A'),
                    mlp_buy_prob=result.get('mlp_buy_prob', 0.0),
                    mlp_hold_prob=result.get('mlp_hold_prob', 0.0),
                    mlp_sell_prob=result.get('mlp_sell_prob', 0.0)
                )
                db_session.merge(ai_score) # Use merge to handle UniqueConstraint if exists
                
                all_results.append(result)
                self.score_ready.emit({ticker: result})
                
                pct = int((i + 1) / len(self.tickers) * 100)
                self.progress.emit(pct)
            
            db_session.commit()
            self.all_done.emit(all_results)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db_session.close()
