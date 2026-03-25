import torch
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# 1. Path & Import Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.dirname(current_dir)
sys.path.append(model_root)

from model import ConformerTransducer
from misc.tokenizer import CharTokenizer

# 2. GPU Detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Config & Model Setup
eeg_config = {
    "input_dim": 14, "encoder_dim": 144, "decoder_dim": 320, 
    "joint_dim": 320, "vocab_size": 8
}
tokenizer = CharTokenizer(transcripts=["aiu", "eon"])

# Move Model to GPU
model = ConformerTransducer(eeg_config).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torchaudio.functional.rnnt_loss

# 4. Dummy Data Setup (Move to GPU)
dummy_eeg = torch.randn(1, 150, 14).to(device)
target_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long).to(device)
in_lens = torch.tensor([150], dtype=torch.int32).to(device)
tgt_lens = torch.tensor([3], dtype=torch.int32).to(device)

# 5. Training Loop
epochs = 5000
loss_history = []

print(f"{'Epoch':<10} | {'Loss':<10}")
print("-" * 25)

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # Prepend blank (0) and ensure it's on the same device
    blank_col = torch.zeros((1, 1), dtype=torch.long, device=device)
    decoder_input = torch.cat([blank_col, target_tokens], dim=1) # No [:-1] slice!
    
    # Forward Pass
    logits = model(dummy_eeg, decoder_input)
    enc_out_lens = model.get_encoder_out_lengths(in_lens)
    
    # Calculate RNN-T Loss
    loss = loss_fn(
        logits=logits,
        targets=target_tokens.to(torch.int32),
        logit_lengths=enc_out_lens.to(torch.int32),
        target_lengths=tgt_lens.to(torch.int32),
        blank=0
    )
    
    loss.backward()
    optimizer.step()
    
    loss_val = loss.item()
    loss_history.append(loss_val)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"{epoch:<10} | {loss_val:<10.4f}")

# 6. Final Visualization
plt.figure(figsize=(8, 5))
plt.plot(loss_history)
plt.title("GPU Accelerated RNN-T Training (EEG)")
plt.show()