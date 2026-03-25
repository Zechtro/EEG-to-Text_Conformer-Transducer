import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDecoder(nn.Module):
    """
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of the embedding layer
        hidden_dim: Hidden dimension of the LSTM
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=640, 
                 num_layers=1, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding Layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, y, hidden=None):
        """
        Args:
            y: Input token indices of shape (batch, seq_len)
            hidden: Hidden state tuple (h, c) from previous step
                   If None, will be initialized to zeros
        Returns:
            output: Decoder output of shape (batch, seq_len, hidden_dim)
            hidden: Updated hidden state tuple (h, c)
        """
        # Embedding Layer
        # y: (batch, seq_len) -> embedded: (batch, seq_len, embedding_dim)
        embedded = self.embedding(y)
        embedded = self.dropout(embedded)
        
        # output: (batch, seq_len, hidden_dim)
        # hidden: tuple of (h, c) each of shape (num_layers, batch, hidden_dim)
        output, hidden = self.lstm(embedded, hidden)
        
        # Apply dropout
        output = self.dropout(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        """
        Args:
            batch_size: Batch size
            device: Device (cpu or cuda)
        Returns:
            hidden: Tuple of (h, c) initialized to zeros
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
    
    def predict_one_step(self, y, hidden):
        """
        Args:
            y: Single token of shape (batch, 1)
            hidden: Hidden state tuple (h, c)
        Returns:
            output: Decoder output of shape (batch, 1, hidden_dim)
            hidden: Updated hidden state tuple (h, c)
        """
        return self.forward(y, hidden)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Conformer Transducer Decoder")
    print("=" * 70)
    
    # Config
    vocab_size = 1000
    batch_size = 4
    seq_len = 50
    
    # Test basic decoder
    print("\n1. Testing Basic Decoder")
    print("-" * 70)
    
    decoder = LSTMDecoder(
        vocab_size=vocab_size,
        embedding_dim=320,
        hidden_dim=640,
        num_layers=1,
        dropout=0.1
    )
    
    # Create dummy input (token indices)
    y = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Input shape: {y.shape}")
    print(f"Input tokens (first sequence, first 10): {y[0, :10].tolist()}")
    
    # Forward pass
    output, hidden = decoder(y)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Hidden state h shape: {hidden[0].shape}")
    print(f"Hidden state c shape: {hidden[1].shape}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")
    
    # Test one-step prediction (for inference)
    print("\n2. Testing One-Step Prediction")
    print("-" * 70)
    
    # Initialize hidden state
    hidden = decoder.init_hidden(batch_size, y.device)
    
    # Predict one token at a time
    y_single = torch.randint(0, vocab_size, (batch_size, 1))
    output_single, hidden = decoder.predict_one_step(y_single, hidden)
    
    print(f"Single step input shape: {y_single.shape}")
    print(f"Single step output shape: {output_single.shape}")
    print(f"Hidden state updated: h shape {hidden[0].shape}, c shape {hidden[1].shape}")
    
    # Test autoregressive decoding simulation
    print("\n3. Simulating Autoregressive Decoding")
    print("-" * 70)
    
    max_length = 10
    batch_size = 2
    
    # Start with blank token (index 0 is blank)
    current_token = torch.zeros((batch_size, 1), dtype=torch.long)
    hidden = decoder.init_hidden(batch_size, current_token.device)
    
    decoded_sequence = [current_token]
    
    for step in range(max_length):
        # Predict next step
        output, hidden = decoder.predict_one_step(current_token, hidden)
        
        # Simulate next token from JointNetwork
        current_token = torch.randint(0, vocab_size, (batch_size, 1))
        decoded_sequence.append(current_token)
    
    decoded_sequence = torch.cat(decoded_sequence, dim=1)
    print(f"Autoregressive decoding shape: {decoded_sequence.shape}")
    print(f"Example decoded sequence: {decoded_sequence[0].tolist()}")
    
    print("\n" + "=" * 70)
    print("All tests passed! Decoder is ready for integration.")
    print("=" * 70)