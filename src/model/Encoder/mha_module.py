import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Multi-Head Self Attention Module
class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Attention Components
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Relative positional encoding
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # LayerNorm
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        k = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        v = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose: (batch, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Relative positional encoding
        pos_emb = self._get_relative_position_encoding(seq_len, x.device)
        pos_emb = self.linear_pos(pos_emb)
        pos_emb = pos_emb.view(seq_len, self.num_heads, self.d_k).transpose(0, 1)
        
        # Self Attention
        q_with_bias_u = q + self.pos_bias_u.unsqueeze(1)
        q_with_bias_v = q + self.pos_bias_v.unsqueeze(1)
        content_score = torch.matmul(q_with_bias_u, k.transpose(2, 3))
        
        pos_score = torch.matmul(q_with_bias_v, pos_emb.unsqueeze(0).transpose(2, 3))
        pos_score = self._relative_shift(pos_score)
        
        scores = (content_score + pos_score) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Dropout
        return self.dropout(self.linear_out(context))
    
    # Relative Positional Encoding
    def _get_relative_position_encoding(self, length, device):
        pos = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device) * 
                            -(math.log(10000.0) / self.d_model))
        
        pe = torch.zeros(length, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe
    
    # Shift Relative Position Score
    def _relative_shift(self, pos_score):
        batch_size, num_heads, seq_len1, seq_len2 = pos_score.size()
        zeros = torch.zeros((batch_size, num_heads, seq_len1, 1), 
                           device=pos_score.device, dtype=pos_score.dtype)
        padded = torch.cat([zeros, pos_score], dim=-1)
        
        padded = padded.view(batch_size, num_heads, seq_len2 + 1, seq_len1)
        shifted = padded[:, :, 1:].view_as(pos_score)
        
        return shifted