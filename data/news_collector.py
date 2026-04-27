import feedparser
import pandas as pd
import yfinance as yf
from datetime import datetime
from sqlalchemy.orm import Session
from db.models import NewsSentiment
from models.finbert_scorer import FinBERTScorer

class NewsCollector:
    RSS_URLS = {
        'yahoo': 'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US',
        'reuters': 'https://feeds.reuters.com/reuters/businessNews',
    }

    def collect_news(self, ticker: str, max_items: int = 8) -> list[dict]:
        """Fetch news for ticker using yfinance"""
        try:
            ticker_obj = yf.Ticker(ticker)
            raw_news = ticker_obj.news
            if not raw_news:
                return []
                
            news_items = []
            for entry in raw_news[:max_items]:
                # Handle both new yfinance format and old format
                content = entry.get('content', {})
                
                headline = content.get('title', entry.get('title', ''))
                summary_raw = content.get('summary', entry.get('summary', entry.get('description', '')))
                
                source = 'Yahoo Finance'
                if 'provider' in content:
                    source = content['provider'].get('displayName', 'Yahoo Finance')
                elif 'publisher' in entry:
                    source = entry.get('publisher', 'Yahoo Finance')
                
                url = entry.get('link', '')
                if not url and 'canonicalUrl' in content:
                    url = content['canonicalUrl'].get('url', '')
                
                published = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                if 'pubDate' in content:
                    published = content['pubDate']
                elif 'providerPublishTime' in entry:
                    ts = entry['providerPublishTime']
                    published = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

                import re as _re
                clean = _re.sub(r'<[^>]+>', '', summary_raw).strip()
                summary = clean[:180] + '...' if len(clean) > 180 else clean

                item = {
                    'headline': headline,
                    'summary': summary,
                    'source': source,
                    'url': url,
                    'published': published
                }
                news_items.append(item)
            return news_items
        except Exception as e:
            print(f"Error collecting news for {ticker}: {e}")
            return []

    def collect_and_score(self, ticker: str, scorer: FinBERTScorer, db_session: Session, max_items: int = 5):
        """Collect news, score each headline, and save to NewsSentiment table"""
        news_list = self.collect_news(ticker, max_items)
        scored_list = []
        
        for item in news_list:
            try:
                # Score headline
                sentiment = scorer.score_headline(item['headline'])
                item['score'] = sentiment['score']
                item['label'] = sentiment['label']
            except Exception as e:
                print(f"Error scoring news for {ticker}: {e}")
                item['score'] = 0.0
                item['label'] = 'neutral'
            
            scored_list.append(item)
            
            # Save to DB
            news_sentiment = NewsSentiment(
                ticker=ticker,
                date=datetime.now().date(),
                headline=item['headline'],
                source=item['source'],
                score=item['score'],
                label=item['label']
            )
            db_session.add(news_sentiment)
        
        try:
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            print(f"Error saving news sentiment to DB: {e}")
            
        return scored_list
