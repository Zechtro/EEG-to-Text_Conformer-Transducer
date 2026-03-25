import torch.nn as nn
from convolution_module import ConvolutionModule
from mha_module import MultiHeadedSelfAttentionModule
from ff_module import FeedForwardModule

class ConformerBlock(nn.Module):
    """
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        conv_kernel_size: Kernel size for convolution module
        dropout: Dropout rate
        ffn_expansion_factor: Expansion factor for feed-forward modules
    """
    def __init__(self, d_model, num_heads, conv_kernel_size=32, 
                 dropout=0.1, ffn_expansion_factor=4):
        super().__init__()
        
        self.ffn1 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        self.self_attn = MultiHeadedSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ffn2 = FeedForwardModule(d_model, ffn_expansion_factor, dropout)
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # FFN with half-step residual
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-headed self-attention
        x = x + self.self_attn(x, mask)
        
        # Convolution module
        x = x + self.conv(x)
        
        # FFN with half-step residual
        x = x + 0.5 * self.ffn2(x)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x