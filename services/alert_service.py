import random
import numpy as np
from datetime import datetime, date, timedelta
from sqlalchemy import desc, func
from db.models import Watchlist, AIScore, AlertHistory

class AlertService:
    THRESHOLD = 15.0

    def get_watchlist(self, db_session) -> list[dict]:
        # Query all items in watchlist
        watchlist = db_session.query(Watchlist).all()
        
        if not watchlist:
            return []

        output = []
        for item in watchlist:
            # Get two most recent scores for this ticker
            scores = db_session.query(AIScore).filter(
                AIScore.ticker == item.ticker
            ).order_by(desc(AIScore.date)).limit(2).all()
            
            curr_score = None
            delta = 0.0
            mlp_signal = "N/A"
            
            if len(scores) >= 1:
                curr_score = scores[0].ensemble_score
                mlp_signal = scores[0].mlp_signal
                
                if len(scores) >= 2:
                    prev_score = scores[1].ensemble_score
                    if curr_score is not None and prev_score is not None:
                        delta = curr_score - prev_score

            output.append({
                "ticker": item.ticker,
                "added_at": item.added_at.isoformat() if item.added_at else None,
                "alert_threshold": item.alert_threshold,
                "current_score": curr_score,
                "delta": delta,
                "mlp_signal": mlp_signal
            })
        return output

    def add_ticker(self, ticker: str, db_session, threshold: float = 15.0):
        # Check if already exists
        exists = db_session.query(Watchlist).filter(Watchlist.ticker == ticker).first()
        if not exists:
            new_item = Watchlist(ticker=ticker, alert_threshold=threshold)
            db_session.add(new_item)
            db_session.commit()

    def remove_ticker(self, ticker: str, db_session):
        db_session.query(Watchlist).filter(Watchlist.ticker == ticker).delete()
        db_session.commit()

    def check_alerts(self, db_session) -> list[dict]:
        # Get latest 2 dates for each ticker in watchlist
        watchlist = db_session.query(Watchlist).all()
        alerts = []

        for item in watchlist:
            # Get two most recent scores
            scores = db_session.query(AIScore).filter(
                AIScore.ticker == item.ticker
            ).order_by(desc(AIScore.date)).limit(2).all()

            if len(scores) < 2:
                continue
            
            curr_score = scores[0].ensemble_score
            prev_score = scores[1].ensemble_score
            delta = curr_score - prev_score

            if abs(delta) >= item.alert_threshold:
                alert_data = {
                    'ticker': item.ticker,
                    'prev_score': prev_score,
                    'curr_score': curr_score,
                    'delta': delta
                }
                alerts.append(alert_data)
                self.save_alert(item.ticker, prev_score, curr_score, delta, db_session)
        
        return alerts

    def save_alert(self, ticker: str, prev: float, curr: float, delta: float, db_session):
        alert = AlertHistory(
            ticker=ticker,
            score_before=prev,
            score_after=curr,
            delta=delta
        )
        db_session.add(alert)
        db_session.commit()

    def get_watchlist_with_mock_scores(self, db_session) -> list[dict]:
        items = db_session.query(Watchlist).all()
        result = []
        for item in items:
            import random
            score = random.uniform(40, 90)
            result.append({
                'ticker': item.ticker,
                'current_score': score,
                'delta': random.uniform(-20, 20),
                'added_at': item.added_at.isoformat() if item.added_at else '',
                'alert_threshold': item.alert_threshold
            })
        return result
