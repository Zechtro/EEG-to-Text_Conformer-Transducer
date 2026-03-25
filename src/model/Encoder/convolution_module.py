import torch
import torch.nn as nn
from swish_activation import Swish

# Gated Linear Unit Activation
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * torch.sigmoid(gate)

# Convolution Module
class ConvolutionModule(nn.Module):
    """
    Args:
        d_model: Model dimension
        kernel_size: Kernel size for depthwise convolution
        dropout: Dropout rate
    """
    def __init__(self, d_model, kernel_size=32, dropout=0.1):
        super().__init__()
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise Convolution with expansion factor 2
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        
        # GLU Activation
        self.glu = GLU(dim=1)
        
        # 1D Depthwise Convolution
        # For even kernel sizes, we use asymmetric padding
        if kernel_size % 2 == 1:
            # Odd kernel size: symmetric padding
            padding = (kernel_size - 1) // 2
            self.pad = None
        else:
            # Even kernel size: we'll manually pad
            padding = 0
            # Left padding is kernel_size // 2 - 1
            # Right padding is kernel_size // 2
            self.pad = (kernel_size // 2 - 1, kernel_size // 2)
        
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            stride=1, padding=padding, groups=d_model
        )
        
        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        
        # Pointwise Convolution
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, time, d_model)
        # LayerNorm
        x = self.layer_norm(x)
        
        # Transpose for Conv1d: (batch, d_model, time)
        x = x.transpose(1, 2)
        
        # Pointwise Convolution
        x = self.pointwise_conv1(x)
        
        # GLU Activation
        x = self.glu(x)
        
        # Manual padding for even kernel sizes
        if self.pad is not None:
            x = torch.nn.functional.pad(x, self.pad)
        
        # 1D Depthwise Convolution
        x = self.depthwise_conv(x)
        
        # BatchNorm
        x = self.batch_norm(x)
        
        # Swish Activation
        x = self.swish(x)
        
        # Pointwise Convolution
        x = self.pointwise_conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Transpose back: (batch, time, d_model)
        return x.transpose(1, 2)