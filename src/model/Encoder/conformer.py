import torch
import torch.nn as nn
from conv_subsampling import ConvSubsampling
from conformer_block import ConformerBlock

class Conformer(nn.Module):
    """
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of Conformer blocks
        conv_kernel_size: Kernel size for convolution modules
        dropout: Dropout rate
        ffn_expansion_factor: Expansion factor for feed-forward modules
    """
    def __init__(self, input_dim=80, d_model=256, num_heads=4, num_layers=16,
                 conv_kernel_size=32, dropout=0.1, ffn_expansion_factor=4):
        super().__init__()
        
        self.subsampling = ConvSubsampling(input_dim, d_model)
        
        self.linear = nn.Linear(self.subsampling.get_out_dim(), d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel_size, dropout, ffn_expansion_factor)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, time, input_dim)
            mask: Optional mask tensor
            
        Returns:
            Output tensor of shape (batch, time//4, d_model)
        """
        # Subsampling
        x = self.subsampling(x)
        x = self.linear(x)
        x = self.dropout(x)
        
        # Process through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x, mask)
        
        return x


# Example usage
if __name__ == "__main__":
    # Small model configuration (10.3M params)
    model_small = Conformer(
        input_dim=80,
        d_model=144,
        num_heads=4,
        num_layers=16,
        conv_kernel_size=32,
        dropout=0.1
    )
    
    # Medium model configuration (30.7M params)
    model_medium = Conformer(
        input_dim=80,
        d_model=256,
        num_heads=4,
        num_layers=16,
        conv_kernel_size=32,
        dropout=0.1
    )
    
    # Large model configuration (118.8M params)
    model_large = Conformer(
        input_dim=80,
        d_model=512,
        num_heads=8,
        num_layers=17,
        conv_kernel_size=32,
        dropout=0.1
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 200
    input_dim = 80
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing small model...")
    output_small = model_small(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_small.shape}")
    print(f"Small model parameters: {sum(p.numel() for p in model_small.parameters()) / 1e6:.2f}M")
    
    print("\nTesting medium model...")
    output_medium = model_medium(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_medium.shape}")
    print(f"Medium model parameters: {sum(p.numel() for p in model_medium.parameters()) / 1e6:.2f}M")
    
    print("\nTesting large model...")
    output_large = model_large(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_large.shape}")
    print(f"Large model parameters: {sum(p.numel() for p in model_large.parameters()) / 1e6:.2f}M")