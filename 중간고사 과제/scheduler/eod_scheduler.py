import json
import os
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
import time

class EODScheduler:
    def __init__(self, db_session, alert_service, config_path='config.json'):
        self.db_session = db_session
        self.alert_service = alert_service
        self.config_path = config_path
        
        # Default config
        self.config = {
            'eod_enabled': True,
            'eod_hour': 18,
            'eod_minute': 30,
            'sp500_count': 100,
            'eod_workers': 4,
            'fred_api_key': None
        }
        self._load_config()
        
        self.scheduler = BackgroundScheduler()

    def _load_config(self):
        from app_paths import get_config_path
        config_file = str(get_config_path())
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
                print(f"[{datetime.now()}] Loaded config from {config_file}")
            except Exception as e:
                print(f"[{datetime.now()}] Error loading config: {e}. Using defaults.")

    def start(self):
        if not self.config.get('eod_enabled', True):
            print(f"[{datetime.now()}] EOD Scheduler is disabled in config.")
            return

        hour = self.config.get('eod_hour', 18)
        minute = self.config.get('eod_minute', 30)
        
        self.scheduler.add_job(
            self._run_eod_batch,
            'cron',
            hour=hour,
            minute=minute,
            id='eod_refresh_job'
        )
        self.scheduler.start()
        print(f"[{datetime.now()}] EOD Scheduler started. Job set for {hour:02d}:{minute:02d} daily.")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            print(f"[{datetime.now()}] EOD Scheduler stopped.")

    def _run_eod_batch(self):
        print(f"\n[{datetime.now()}] >>> STARTING EOD BATCH PIPELINE <<<")
        
        # Step 1: Collect OHLCV
        try:
            print(f"[{datetime.now()}] Step 1: Collecting OHLCV data...")
            from data.collector import DataCollector
            collector = DataCollector()
            all_tickers = collector.get_sp500_tickers()
            count = self.config.get('sp500_count', 100)
            tickers = all_tickers[:count]
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Get last 60 days to ensure enough data for indicators
            start_date = (datetime.now().replace(day=1) - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
            
            collector.collect_ohlcv(tickers, start=start_date, end=end_date, db_session=self.db_session)
            print(f"[{datetime.now()}] Step 1 Completed: Collected data for {len(tickers)} tickers.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 1 Failed: {e}")

        # Step 2: FRED exchange rate update (USD/KRW)
        try:
            print(f"[{datetime.now()}] Step 2: Updating exchange rates...")
            from data.collector import DataCollector
            collector = DataCollector()
            # The DataCollector.collect_fred_rates uses yfinance fallback as per its implementation
            collector.collect_fred_rates(self.db_session)
            print(f"[{datetime.now()}] Step 2 Completed.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 2 Failed: {e}")

        # Step 3: Recalculate technical indicators
        try:
            print(f"[{datetime.now()}] Step 3: Recalculating technical indicators...")
            from data.preprocessor import Preprocessor
            preprocessor = Preprocessor()
            # Reuse tickers from step 1
            preprocessor.process_all_tickers(tickers, self.db_session)
            print(f"[{datetime.now()}] Step 3 Completed.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 3 Failed: {e}")

        # Step 4: Regenerate CNN candle images
        try:
            print(f"[{datetime.now()}] Step 4: Regenerating CNN candle images...")
            from data.image_renderer import ImageRenderer
            renderer = ImageRenderer()
            renderer.batch_render(tickers, self.db_session)
            print(f"[{datetime.now()}] Step 4 Completed.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 4 Failed: {e}")

        # Step 5: RSS news + FinBERT sentiment
        try:
            print(f"[{datetime.now()}] Step 5: Collecting and scoring news sentiment...")
            from data.news_collector import NewsCollector
            from models.finbert_scorer import FinBERTScorer
            news_collector = NewsCollector()
            scorer = FinBERTScorer()
            # Only process top 20 tickers for news to save time/resources in batch
            for ticker in tickers[:20]:
                news_collector.collect_and_score(ticker, scorer, self.db_session)
            print(f"[{datetime.now()}] Step 5 Completed.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 5 Failed: {e}")

        # Step 6: Batch AI inference
        try:
            print(f"[{datetime.now()}] Step 6: Running batch AI inference...")
            # Use run_inference script if it exists
            script_path = os.path.join('scripts', 'run_inference.py')
            if os.path.exists(script_path):
                from scripts.run_inference import run_inference
                run_inference()
                print(f"[{datetime.now()}] Step 6 Completed.")
            else:
                print(f"[{datetime.now()}] Step 6 Skipped: scripts/run_inference.py not found.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 6 Failed: {e}")

        # Step 7: Save ensemble scores to DB
        # This is already handled by run_inference() in Step 6
        print(f"[{datetime.now()}] Step 7: Ensemble scores saved (handled in Step 6).")

        # Step 8: Check alert thresholds
        try:
            print(f"[{datetime.now()}] Step 8: Checking alert thresholds...")
            self.alert_service.check_alerts(self.db_session)
            print(f"[{datetime.now()}] Step 8 Completed.")
        except Exception as e:
            print(f"[{datetime.now()}] Step 8 Failed: {e}")

        print(f"[{datetime.now()}] >>> EOD BATCH PIPELINE COMPLETED <<<\n")
