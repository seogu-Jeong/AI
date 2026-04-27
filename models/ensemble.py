import os
import torch
import numpy as np
from .lstm_model import LSTMModel
from .cnn_model import CNNModel
from .transformer_model import TransformerFactorModel
from .mlp_model import MLPModel

class EnsembleScorer:
    def __init__(self, weights={'lstm': 0.3, 'cnn': 0.25, 'transformer': 0.25, 'mlp': 0.2}):
        self.weights = weights
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize models
        self.lstm_model = LSTMModel().to(self.device)
        self.cnn_model = CNNModel().to(self.device)
        self.transformer_model = TransformerFactorModel().to(self.device)
        self.mlp_model = MLPModel().to(self.device)
        
        self.load_models()

    def load_models(self):
        # Paths for weights
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from app_paths import get_weights_dir
        base_path = str(get_weights_dir())
        models_map = {
            'lstm_best.pt': self.lstm_model,
            'cnn_best.pt': self.cnn_model,
            'transformer_best.pt': self.transformer_model,
            'mlp_best.pt': self.mlp_model
        }
        
        for filename, model in models_map.items():
            path = os.path.join(base_path, filename)
            if os.path.exists(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    print(f"Loaded {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"Weights for {filename} not found, using initialized weights.")

    def compute_score(self, ticker: str, lstm_seq: np.ndarray, cnn_image: np.ndarray, transformer_factors: np.ndarray) -> dict:
        tf_flat = np.array(transformer_factors).flatten()
        # Run predictions
        lstm_res = self.lstm_model.predict(lstm_seq, self.device)
        cnn_res = self.cnn_model.predict(cnn_image, self.device)
        trans_res = self.transformer_model.predict(tf_flat[:24], self.device)
        
        # Prepare MLP input: concat(trans_factors[24], lstm_pred[3], cnn_pred[3]) -> 30
        # For simplicity in this logic, we assume inputs are correctly sized
        # Re-run inference to get raw probabilities for MLP input
        self.lstm_model.eval()
        self.cnn_model.eval()
        self.transformer_model.eval()
        
        with torch.no_grad():
            l_in = torch.FloatTensor(lstm_seq).to(self.device)
            if l_in.ndim == 2: l_in = l_in.unsqueeze(0)
            p5, _, _ = self.lstm_model(l_in)
            lstm_3vals = p5[0].cpu().numpy() # (3,)
            
            c_in = torch.FloatTensor(cnn_image).to(self.device)
            if c_in.ndim == 3: c_in = c_in.unsqueeze(0)
            elif c_in.shape[1] != 3: c_in = c_in.permute(0, 3, 1, 2) # Assume BHWC -> BCHW
            cnn_3vals = self.cnn_model(c_in / 255.0)[0].cpu().numpy() # (3,)
            
            t_in = torch.FloatTensor(tf_flat[:24]).unsqueeze(0).to(self.device)
            # Take first 24 factors if more are provided
            mlp_feat = np.concatenate([tf_flat[:24], lstm_3vals, cnn_3vals])
            mlp_res = self.mlp_model.predict(mlp_feat, self.device)

        # Ensemble Score calculation (0-100)
        # Using specific 'positive' signals from each model
        lstm_score = lstm_res['up_5d']
        cnn_score = cnn_res['bullish']
        trans_score = trans_res['up_prob']
        mlp_score = mlp_res['buy']
        
        w = self.weights
        ensemble_score = (
            w['lstm'] * lstm_score +
            w['cnn'] * cnn_score +
            w['transformer'] * trans_score +
            w['mlp'] * mlp_score
        ) * 100

        return {
            'ticker': ticker,
            'ensemble_score': float(ensemble_score),
            'lstm_score': float(lstm_score * 100),
            'cnn_score': float(cnn_score * 100),
            'transformer_score': float(trans_score * 100),
            'mlp_score': float(mlp_score * 100),
            'lstm_dir_5d': lstm_res['dir_5d'],
            'lstm_dir_20d': lstm_res['dir_20d'],
            'lstm_dir_60d': lstm_res['dir_60d'],
            'cnn_pattern': cnn_res['pattern'],
            'mlp_signal': mlp_res['signal'],
            'mlp_buy_prob': mlp_res['buy'],
            'mlp_hold_prob': mlp_res['hold'],
            'mlp_sell_prob': mlp_res['sell'],
            'attention_weights': trans_res['attention_weights'],
            'top_factors': trans_res['top_factors']
        }

    def generate_mock_score(self, ticker: str) -> dict:
        # Seed by ticker for deterministic but "realistic" random values
        seed = hash(ticker) % 1000
        np.random.seed(seed)
        
        # Use distributions skewed by ticker name length or just random
        lstm_up = np.random.beta(5, 5)
        cnn_bullish = np.random.beta(2, 2)
        trans_up = np.random.beta(3, 4)
        mlp_buy = np.random.beta(4, 3)
        
        # Mock attention weights
        from .transformer_model import FACTOR_NAMES
        weights = np.random.dirichlet(np.ones(len(FACTOR_NAMES)), size=1)[0]
        factor_impact = sorted(list(zip(FACTOR_NAMES, weights.tolist())), key=lambda x: x[1], reverse=True)
        
        w = self.weights
        ensemble = (w['lstm']*lstm_up + w['cnn']*cnn_bullish + w['transformer']*trans_up + w['mlp']*mlp_buy) * 100
        
        return {
            'ticker': ticker,
            'ensemble_score': float(ensemble),
            'lstm_score': float(lstm_up * 100),
            'cnn_score': float(cnn_bullish * 100),
            'transformer_score': float(trans_up * 100),
            'mlp_score': float(mlp_buy * 100),
            'lstm_dir_5d': 'UP' if lstm_up > 0.4 else 'DOWN',
            'lstm_dir_20d': 'UP' if lstm_up > 0.45 else 'FLAT',
            'lstm_dir_60d': 'UP' if lstm_up > 0.5 else 'DOWN',
            'cnn_pattern': 'Bullish' if cnn_bullish > 0.4 else 'Bearish',
            'mlp_signal': 'Buy' if mlp_buy > 0.5 else 'Hold',
            'mlp_buy_prob': float(mlp_buy),
            'mlp_hold_prob': float(1 - mlp_buy),
            'mlp_sell_prob': 0.0,
            'attention_weights': weights.tolist(),
            'top_factors': factor_impact[:5]
        }

    def update_weights_from_accuracy(self, accuracies: dict):
        # accuracies: {'lstm': 0.8, 'cnn': 0.75, ...}
        # Update via softmax normalization
        keys = list(accuracies.keys())
        vals = np.array([accuracies[k] for k in keys])
        exp_vals = np.exp(vals * 10) # Temperature scaling
        new_weights = exp_vals / np.sum(exp_vals)
        
        for i, k in enumerate(keys):
            self.weights[k] = float(new_weights[i])
        print(f"Updated weights: {self.weights}")
