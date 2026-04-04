#!/usr/bin/env python3
import sys
import torch
sys.path.insert(0, 'src/model')
import torchaudio.functional as F

from model import ConformerTransducer
from misc.tokenizer import CharTokenizer

# Setup
config = {
    'input_dim': 14 * 80,  # 1120
    'encoder_dim': 256,
    'decoder_dim': 512,
    'joint_dim': 512,
    'vocab_size': 100,
}

tokenizer = CharTokenizer(transcripts=['hello', 'world', 'test'])
config['vocab_size'] = tokenizer.vocab_size()
model = ConformerTransducer(config)
device = 'cpu'
model = model.to(device)

# Dummy data (similar to batch shape)
batch_size = 2
dummy_audio = torch.randn(batch_size, 100, 1120)
dummy_targets = torch.randint(1, 10, (batch_size, 5))
feature_length = torch.LongTensor([100, 100])
target_length = torch.LongTensor([5, 5])

print("=" * 70)
print("TESTING SHAPES FOR RNN-T FIX")
print("=" * 70)

# Encoder
encoder_out = model.encoder(dummy_audio)
print(f"\n✓ Encoder output shape: {encoder_out.shape}")
print(f"  Expected: (batch={batch_size}, enc_time, encoder_dim={config['encoder_dim']})")

# Decoder with blank prepended (FIX)
blank_col = torch.zeros((batch_size, 1), dtype=torch.long)
decoder_input = torch.cat([blank_col, dummy_targets], dim=1)
print(f"\n✓ Decoder input shape (with blank): {decoder_input.shape}")
print(f"  Expected: (batch={batch_size}, target_len+1={dummy_targets.shape[1]+1})")

hidden_state = model.decoder.init_hidden(batch_size, device)
decoder_out, _ = model.decoder(decoder_input, hidden_state)
print(f"\n✓ Decoder output shape: {decoder_out.shape}")

# Joiner
enc_proj = model.joiner.encoder_proj(encoder_out)
dec_proj = model.joiner.decoder_proj(decoder_out)
joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
logits = model.joiner.output_proj(joint)
print(f"\n✓ Logits shape: {logits.shape}")
print(f"  Expected: (batch={batch_size}, enc_time={encoder_out.shape[1]}, dec_time={decoder_out.shape[1]}, vocab_size={config['vocab_size']})")

# Test rnnt_loss
print("\nTesting torchaudio.functional.rnnt_loss...")
try:
    # IMPORTANT: Use encoder output lengths (after subsampling)
    enc_out_lengths = model.get_encoder_out_lengths(feature_length)
    print(f"  Input feature_length: {feature_length.tolist()}")
    print(f"  Encoder output lengths (after subsampling): {enc_out_lengths.tolist()}")
    
    loss = F.rnnt_loss(
        logits=logits,
        targets=dummy_targets.to(torch.int32),
        logit_lengths=enc_out_lengths.to(torch.int32),
        target_lengths=target_length.to(torch.int32),
        blank=0
    )
    print(f"✓ RNN-T Loss computed successfully: {loss.item():.4f}")
    print("\n" + "=" * 70)
    print("✓ ALL SHAPES CORRECT! Ready for training.")
    print("=" * 70)
except Exception as e:
    print(f"✗ RNN-T Loss error: {e}")
    import traceback
    traceback.print_exc()
