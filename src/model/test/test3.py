import torch
import torchaudio.functional as F
from indobenchmark import IndoNLGTokenizer
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.dirname(current_dir)
sys.path.append(model_root)

from model import ConformerIndoGPTTransducer
from misc.beam_decoder import BeamDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Setup
tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indogpt")
vocab_size = tokenizer.vocab_size + 1
model = ConformerIndoGPTTransducer({"input_dim": 14, "encoder_dim": 144, "decoder_dim": 320, 
                                    "joint_dim": 320, "vocab_size": vocab_size, "num_layers": 4}).to(device)

# 2. Fixed Data for Overfitting
phrases = ["saya belajar", "makan nasi"]
encoded = [tokenizer.encode(p) for p in phrases]
shifted = [[t + 1 for t in seq] for seq in encoded]
actual_max_len = max(len(s) for s in shifted)
padded_shifted = [s + [0] * (actual_max_len - len(s)) for s in shifted]

target_tokens = torch.tensor(padded_shifted).to(device)
tgt_lens = torch.tensor([len(s) for s in shifted], dtype=torch.int32).to(device)

# CREATE FIXED EEG ONCE
fixed_eeg = torch.randn(2, 200, 14).to(device)

# 3. Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_epoch = 5000
for epoch in range(1, total_epoch+1): # Increased epochs slightly to ensure memorization
    optimizer.zero_grad()
    blank_col = torch.zeros((2, 1), dtype=torch.long, device=device)
    decoder_input = torch.cat([blank_col, target_tokens], dim=1)
    
    logits = model(fixed_eeg, decoder_input)
    logit_lens = model.get_encoder_out_lengths(torch.tensor([200, 200])).to(torch.int32).to(device)
    
    loss = F.rnnt_loss(logits=logits, targets=target_tokens.to(torch.int32),
                       logit_lengths=logit_lens, target_lengths=tgt_lens, blank=0)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 4. Inference on the SAME data
print("\n--- Testing Memorization ---")
beam_decoder = BeamDecoder(model, tokenizer, beam_size=5)

for i, phrase in enumerate(phrases):
    # Slice the exact EEG sample used for this phrase
    sample_eeg = fixed_eeg[i:i+1] 
    result = beam_decoder.decode(sample_eeg)
    print(f"Target: '{phrase}' | Decoded: '{result}'")