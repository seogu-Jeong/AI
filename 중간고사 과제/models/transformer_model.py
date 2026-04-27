import torch
import torch.nn as nn
import numpy as np

FACTOR_NAMES = [
    'RSI(14)','MACD','BB%B','ATR','OBV변화율','거래량비율','Stoch%K','MA크로스',
    'PER','PBR','ROE','EPS성장률','부채비율','영업이익률','시총(log)','52주고가비율',
    '1개월수익률','3개월수익률','USD/KRW변화율','VIX','뉴스감성','뉴스감성5MA',
    '배당수익률','EPS서프라이즈'
]

class TransformerFactorModel(nn.Module):
    def __init__(self, n_factors=24, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerFactorModel, self).__init__()
        # Embedding: treats each factor as a token
        self.embedding = nn.Linear(1, d_model)
        
        # We replace nn.TransformerEncoder with a manual list of layers 
        # to facilitate attention weight extraction.
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=256, 
                dropout=dropout, 
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.fc1 = nn.Linear(d_model, 32)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.n_factors = n_factors
        self._attention_weights = []

        # Register forward hooks on each layer's self_attn (MultiheadAttention)
        for layer in self.layers:
            layer.self_attn.register_forward_hook(self._capture_attention_hook)

    def _capture_attention_hook(self, module, input, output):
        """
        Hook to capture attn_output_weights from nn.MultiheadAttention.
        As nn.TransformerEncoderLayer calls self_attn with need_weights=False,
        we re-run the forward pass of the self_attn module with need_weights=True.
        """
        # input contains (query, key, value)
        query, key, value = input[0], input[1], input[2]
        
        # We call .forward() directly to avoid triggering hooks recursively
        with torch.no_grad():
            _, weights = module.forward(
                query, key, value, 
                need_weights=True, 
                average_attn_weights=True
            )
            self._attention_weights.append(weights)

    def forward(self, x):
        # x shape: (batch, n_factors)
        batch_size = x.shape[0]
        self._attention_weights = [] # Reset weights for each forward pass
        
        # Reshape to (batch, n_factors, 1) to treat each factor as a sequence element
        x = x.unsqueeze(-1)
        
        # Embedding
        x = self.embedding(x) # (batch, n_factors, d_model)
        
        # Transformer: Apply layers manually
        for layer in self.layers:
            x = layer(x)
        
        # Global Average Pool over factors
        pooled = x.mean(dim=1) # (batch, d_model)
        
        # Heads
        logits = self.fc2(self.gelu(self.fc1(pooled)))
        prob = self.sigmoid(logits)
        
        # Extract and average attention weights
        if len(self._attention_weights) > 0:
            # self._attention_weights is a list of tensors of shape (batch, n_factors, n_factors)
            all_layer_weights = torch.stack(self._attention_weights) # (num_layers, batch, n_factors, n_factors)
            avg_weights = all_layer_weights.mean(dim=0) # (batch, n_factors, n_factors)
            
            # Average across the query dimension (dim 1) to get importance per factor
            factor_importance = avg_weights.mean(dim=1) # (batch, n_factors)
            
            # Final normalization with softmax as requested
            attn_weights = torch.softmax(factor_importance, dim=1)
        else:
            # Fallback (should not happen if hooks are working)
            attn_weights = torch.ones(batch_size, self.n_factors).to(x.device) / self.n_factors
        
        return prob, attn_weights

    def predict(self, factors: np.ndarray, device='cpu') -> dict:
        self.eval()
        with torch.no_grad():
            if factors.ndim == 1:
                factors = np.expand_dims(factors, axis=0)
            
            tensor_x = torch.FloatTensor(factors).to(device)
            prob, attn = self.forward(tensor_x)
            
            prob_val = float(prob.cpu().numpy()[0][0])
            attn_list = attn.cpu().numpy()[0].tolist()
            
            # Pair factors with weights
            factor_impact = list(zip(FACTOR_NAMES, attn_list))
            top_factors = sorted(factor_impact, key=lambda x: x[1], reverse=True)
            
            return {
                'up_prob': prob_val,
                'attention_weights': attn_list,
                'top_factors': top_factors
            }
