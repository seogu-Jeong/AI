import torch
import threading

_scorer_instance = None
_scorer_lock = threading.Lock()

def get_finbert_scorer():
    global _scorer_instance
    with _scorer_lock:
        if _scorer_instance is None:
            _scorer_instance = FinBERTScorer()
    return _scorer_instance

class FinBERTScorer:
    def __init__(self):
        self.model_name = 'ProsusAI/finbert'
        self.pipeline = None # Lazy load

    def _load_model(self):
        if self.pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
            except ImportError:
                raise ImportError("The 'transformers' library is required for FinBERTScorer.")
            
            # Load pipeline for text-classification
            self.pipeline = hf_pipeline(
                'text-classification', 
                model=self.model_name, 
                top_k=None,
                device=0 if torch.cuda.is_available() else -1
            )

    def score_headline(self, text: str) -> dict:
        self._load_model()
        results = self.pipeline(text)[0]
        
        # Map labels to scores
        scores = {res['label']: res['score'] for res in results}
        # FinBERT labels are usually 'positive', 'negative', 'neutral'
        pos = scores.get('positive', 0.0)
        neg = scores.get('negative', 0.0)
        neu = scores.get('neutral', 0.0)
        
        # Sentiment score in range [-1, 1]
        sentiment_score = pos - neg
        
        # Determine winning label
        label = max(scores, key=scores.get)
        
        return {
            'score': float(sentiment_score),
            'label': label,
            'positive': float(pos),
            'negative': float(neg),
            'neutral': float(neu)
        }

    def score_headlines(self, headlines: list[str]) -> list[dict]:
        if not headlines:
            return []
        return [self.score_headline(h) for h in headlines]

    def get_avg_sentiment(self, headlines: list[str]) -> float:
        if not headlines:
            return 0.0
        results = self.score_headlines(headlines)
        scores = [r['score'] for r in results]
        return sum(scores) / len(scores)
