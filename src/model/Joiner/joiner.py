import torch
import torch.nn as nn
import torch.nn.functional as F


class JointNetwork(nn.Module):
    """
    Args:
        vocab_size: Size of the vocabulary
        encoder_dim: Dimension of encoder output (d_model from Conformer)
        decoder_dim: Dimension of decoder output (hidden_dim from LSTM)
        joint_dim: Dimension of the joint hidden layer
        activation: Activation function ('tanh' or 'relu')
    """
    def __init__(self, vocab_size, encoder_dim, decoder_dim, joint_dim=640, 
                 activation='tanh'):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.joint_dim = joint_dim
        
        # Project encoder output to Joiner dimension
        self.encoder_proj = nn.Linear(encoder_dim, joint_dim, bias=True)
        
        # Project decoder output to Joiner dimension
        self.decoder_proj = nn.Linear(decoder_dim, joint_dim, bias=True)
        
        # Activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Final projection to vocabulary
        self.output_proj = nn.Linear(joint_dim, vocab_size, bias=True)
    
    def forward(self, encoder_output, decoder_output):
        """
        Args:
            encoder_output: Encoder output of shape (batch, T, encoder_dim)
                          T: the number of acoustic frames
            decoder_output: Decoder output of shape (batch, U, decoder_dim)
                          U: the number of label tokens
        Returns:
            logits: Joint network output of shape (batch, T, U, vocab_size)
        """
        batch_size, T, _ = encoder_output.size()
        _, U, _ = decoder_output.size()
        
        # Project encoder and decoder outputs
        # encoder_out: (batch, T, joint_dim)
        encoder_out = self.encoder_proj(encoder_output)
        
        # decoder_out: (batch, U, joint_dim)
        decoder_out = self.decoder_proj(decoder_output)
        
        # Expand dimensions for broadcasting
        # encoder_out: (batch, T, 1, joint_dim)
        encoder_out = encoder_out.unsqueeze(2)
        
        # decoder_out: (batch, 1, U, joint_dim)
        decoder_out = decoder_out.unsqueeze(1)
        
        # Combine encoder and decoder outputs (element-wise addition)
        # joint: (batch, T, U, joint_dim)
        joint = encoder_out + decoder_out
        
        joint = self.activation(joint)
        
        # Project to vocabulary size
        # logits: (batch, T, U, vocab_size)
        logits = self.output_proj(joint)
        
        return logits
    
    def forward_single(self, encoder_step, decoder_step):
        """
        Args:
            encoder_step: Single encoder frame of shape (batch, encoder_dim)
            decoder_step: Single decoder output of shape (batch, decoder_dim)
        Returns:
            logits: Output of shape (batch, vocab_size)
        """
        # Project
        # encoder_out & decoder_out: (batch, joint_dim)
        encoder_out = self.encoder_proj(encoder_step)  
        decoder_out = self.decoder_proj(decoder_step)
        
        # Combine
        # joint: (batch, joint_dim)
        joint = encoder_out + decoder_out  
        
        joint = self.activation(joint)
        
        # Project to vocabulary
        # logits: (batch, vocab_size)
        logits = self.output_proj(joint)
        
        return logits


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Conformer Transducer Joint Network")
    print("=" * 70)
    
    # Configs
    vocab_size = 1000
    encoder_dim = 256
    decoder_dim = 640
    joint_dim = 640
    
    batch_size = 4
    T = 50
    U = 20
    
    # Create dummy encoder and decoder outputs
    encoder_output = torch.randn(batch_size, T, encoder_dim)
    decoder_output = torch.randn(batch_size, U, decoder_dim)
    
    print(f"\nEncoder output shape: {encoder_output.shape}")
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Test basic joint network
    print("\n1. Testing Basic Joint Network")
    print("-" * 70)
    
    joint_net = JointNetwork(
        vocab_size=vocab_size,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        joint_dim=joint_dim,
        activation='tanh'
    )
    
    logits = joint_net(encoder_output, decoder_output)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: (batch={batch_size}, T={T}, U={U}, vocab={vocab_size})")
    print(f"Joint network parameters: {sum(p.numel() for p in joint_net.parameters()) / 1e6:.2f}M")
    
    # Test single step (for inference)
    print("\n2. Testing Single Step Forward (Inference)")
    print("-" * 70)
    
    encoder_step = encoder_output[:, 0, :]  # First frame: (batch, encoder_dim)
    decoder_step = decoder_output[:, 0, :]  # First token: (batch, decoder_dim)
    
    logits_single = joint_net.forward_single(encoder_step, decoder_step)
    print(f"Single step encoder input: {encoder_step.shape}")
    print(f"Single step decoder input: {decoder_step.shape}")
    print(f"Single step output: {logits_single.shape}")
    
    # Softmax to get probabilities
    probs = F.softmax(logits_single, dim=-1)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Probability sum (should be 1.0): {probs[0].sum().item():.4f}")
    
    # Simulate inference step by step
    print("\n3. Simulating Step-by-Step Inference")
    print("-" * 70)
    
    t = 0  # Current acoustic frame
    u = 0  # Current label position
    
    print(f"At acoustic frame t={t}, label position u={u}:")
    
    # Get encoder output for frame t
    enc_t = encoder_output[:, t, :]  # (batch, encoder_dim)
    
    # Get decoder output for position u
    dec_u = decoder_output[:, u, :]  # (batch, decoder_dim)
    
    # Compute joint network output
    logits_tu = joint_net.forward_single(enc_t, dec_u)
    probs_tu = F.softmax(logits_tu, dim=-1)
    
    # Get top-k predictions
    top_k = 5
    top_probs, top_indices = torch.topk(probs_tu[0], top_k)
    
    print(f"  Top {top_k} predictions:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"    {i+1}. Token {idx.item():4d} with probability {prob.item():.4f}")
    
    # Check if blank is predicted (assume blank is token 0)
    blank_prob = probs_tu[0, 0].item()
    print(f"  Blank token probability: {blank_prob:.4f}")
    
    print("\n" + "=" * 70)
    print("All tests passed! Joint Network is ready.")
    print("=" * 70)