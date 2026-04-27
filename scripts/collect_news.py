import os
import sys
from datetime import datetime, timedelta
from sqlalchemy import func

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import SessionLocal
from db.models import RawOHLCV, NewsSentiment, Feature
from data.news_collector import NewsCollector
from models.finbert_scorer import FinBERTScorer

def main():
    print("Starting news collection and sentiment scoring...")
    db = SessionLocal()
    
    try:
        # 1. Get all distinct tickers from RawOHLCV
        tickers = [r[0] for r in db.query(RawOHLCV.ticker).distinct().all()]
        total = len(tickers)
        print(f"Found {total} tickers to process.")

        # 2. Initialize Scorer and Collector
        # This will download the model if not present on first use
        scorer = FinBERTScorer()
        collector = NewsCollector()

        today = datetime.now().date()
        
        for i, ticker in enumerate(tickers, 1):
            try:
                # a. Delete existing news_sentiment rows for this ticker and today
                db.query(NewsSentiment).filter(
                    NewsSentiment.ticker == ticker,
                    NewsSentiment.date == today
                ).delete()
                db.commit()

                # b. Collect and score news
                # max_items=5 as requested
                scored_items = collector.collect_and_score(ticker, scorer, db, max_items=5)
                
                n_articles = len(scored_items)
                avg_score = 0.0
                if n_articles > 0:
                    avg_score = sum(item['score'] for item in scored_items) / n_articles

                # c. Calculate 5-day rolling average
                five_days_ago = today - timedelta(days=4)
                all_recent_scores = db.query(NewsSentiment.score).filter(
                    NewsSentiment.ticker == ticker,
                    NewsSentiment.date >= five_days_ago
                ).all()
                
                if all_recent_scores:
                    scores_list = [s[0] for s in all_recent_scores]
                    avg_5d = sum(scores_list) / len(scores_list)
                else:
                    avg_5d = avg_score

                # d. Update Feature table: set news_sentiment for the most recent feature row
                latest_feat = db.query(Feature).filter(
                    Feature.ticker == ticker
                ).order_by(Feature.date.desc()).first()

                if latest_feat:
                    latest_feat.news_sentiment = avg_score
                    latest_feat.news_sentiment_5d = avg_5d
                    db.commit()
                
                print(f"[{i}/{total}] {ticker}: score={avg_score:.3f}, {n_articles} articles")

            except Exception as ticker_err:
                db.rollback()
                print(f"[{i}/{total}] {ticker}: FAILED - {ticker_err}")

        print("\n" + "="*40)
        print("News collection summary:")
        print(f"Successfully processed {total} tickers.")
        print("="*40)

    finally:
        db.close()

if __name__ == "__main__":
    main()
