import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_src = os.path.dirname(current_dir) 
sys.path.append(os.path.join(root_src))

from model import ConformerTransducer
from misc.tokenizer import CharTokenizer

import torch
import torchaudio
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Config & Model Setup
eeg_config = {
    "input_dim": 14, "encoder_dim": 144, "decoder_dim": 320, 
    "joint_dim": 320, "vocab_size": 8
}
tokenizer = CharTokenizer(transcripts=["aiu", "eon"])
model = ConformerTransducer(eeg_config)
optimizer = optim.Adam(model.parameters(), lr=0.001) # Higher LR for quick test
loss_fn = torchaudio.functional.rnnt_loss

# 2. Static Dummy Data (to see if model can "overfit"/learn one sample)
dummy_eeg = torch.randn(1, 150, 14) # [Batch, Time, Channels]
target_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long) # [Batch, Seq]
in_lens = torch.tensor([150], dtype=torch.int32)
tgt_lens = torch.tensor([3], dtype=torch.int32)

# 3. Training Loop
epochs = 10000
loss_history = []

print(f"{'Epoch':<10} | {'Loss':<10}")
print("-" * 25)

model.train()
for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    
    # 1. DO NOT SLICE. Use the full sequence: [blank, token1, token2, token3]
    # This ensures decoder_out has length U = 4
    decoder_input = torch.cat([torch.zeros((1, 1), dtype=torch.long), target_tokens], dim=1)
    
    # 2. Forward Pass
    logits = model(dummy_eeg, decoder_input)

    # 3. Get lengths
    enc_out_lens = model.get_encoder_out_lengths(in_lens)

    # 4. Compute Loss
    # Now logits.shape[2] will be 4, which matches tgt_lens + 1
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
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:<10} | {loss_val:<10.4f}")

# 4. Visualization
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, marker='o', color='b', label='Training Loss')
plt.title("RNN-T Loss Convergence Test (EEG-to-Text)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()