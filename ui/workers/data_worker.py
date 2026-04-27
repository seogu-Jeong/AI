import datetime
from PySide6.QtCore import QThread, Signal
from data.collector import DataCollector
from data.preprocessor import Preprocessor
from data.image_renderer import ImageRenderer
from db.database import SessionLocal
from db.models import RawOHLCV
from sqlalchemy import select
import pandas as pd

class DataWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, tickers: list[str] = None):
        super().__init__()
        if tickers is None:
            self.tickers = DataCollector.get_cached_sp500_tickers()
        else:
            self.tickers = tickers
        
        self.collector = DataCollector()
        self.preprocessor = Preprocessor()
        self.renderer = ImageRenderer()

    def run(self):
        db_session = SessionLocal()
        try:
            # Step 1: Emit status('데이터 수집 중...'), collect OHLCV via DataCollector
            self.status.emit('데이터 수집 중...')
            self.progress.emit(0)
            
            end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            
            # Read history_years from config.json (default 3 if not found)
            import json, os
            from app_paths import get_config_path, get_user_file
            history_years = 3
            try:
                config_path = get_config_path()
                if config_path.exists():
                    with open(str(config_path), 'r') as f:
                        cfg = json.load(f)
                        history_years = int(cfg.get('history_years', 3))
            except Exception:
                pass
                
            start_date = (datetime.datetime.now() - datetime.timedelta(days=365 * history_years)).strftime('%Y-%m-%d')
            
            self.collector.collect_ohlcv(self.tickers, start=start_date, end=end_date, db_session=db_session)
            self.progress.emit(33)
            
            if self.isInterruptionRequested(): return

            # Step 2: status('기술적 지표 계산 중...'), run preprocessor
            self.status.emit('기술적 지표 계산 중...')
            self.preprocessor.process_all_tickers(self.tickers, db_session)
            self.progress.emit(66)

            if self.isInterruptionRequested(): return

            # Step 3: status('이미지 생성 중...'), run image_renderer
            self.status.emit('이미지 생성 중...')
            # Create cache dir if not exists
            import os
            cache_dir = str(get_user_file('cache/images'))
            os.makedirs(cache_dir, exist_ok=True)
            
            for i, ticker in enumerate(self.tickers):
                if self.isInterruptionRequested(): return
                
                # Fetch data for rendering
                stmt = select(RawOHLCV).where(RawOHLCV.ticker == ticker).order_by(RawOHLCV.date.desc()).limit(100)
                rows = db_session.execute(stmt).scalars().all()
                if rows:
                    data = [{
                        'date': r.date,
                        'open': r.open,
                        'high': r.high,
                        'low': r.low,
                        'close': r.close,
                        'volume': r.volume
                    } for r in reversed(rows)]
                    df = pd.DataFrame(data)
                    df.set_index('date', inplace=True)
                    
                    output_path = os.path.join(cache_dir, f"{ticker}.png")
                    self.renderer.render_candle_image(df, ticker, output_path=output_path)
                
                # Intermediate progress for rendering
                total_tickers = len(self.tickers)
                if total_tickers > 0:
                    render_progress = 66 + int((i + 1) / total_tickers * 34)
                    self.progress.emit(min(render_progress, 100))

            self.progress.emit(100)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
        finally:
            db_session.close()

    def stop(self):
        self.requestInterruption()
