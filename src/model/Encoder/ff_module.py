import torch.nn as nn
from swish_activation import Swish

class FeedForwardModule(nn.Module):
    """
    Args:
        d_model: Model dimension
        expansion_factor: Expansion factor for the hidden dimension
        dropout: Dropout rate
    """
    def __init__(self, d_model, expansion_factor=4, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)