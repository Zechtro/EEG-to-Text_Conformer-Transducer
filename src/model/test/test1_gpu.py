import torch
import torchaudio
import torch.optim as optim
from indobenchmark import IndoNLGTokenizer
import matplotlib.pyplot as plt
import sys
import os

# 1. Path & Import Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
model_root = os.path.dirname(current_dir)
sys.path.append(model_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GPU MONITORING FUNCTION ---
def print_gpu_utilization():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        print(f"   [GPU] Allocated: {allocated:.2f}MB | Reserved: {reserved:.2f}MB")

# --- VISUALIZATION FUNCTION ---
def plot_alignment(logits, phrase_tokens, epoch):
    activation_map = torch.max(logits, dim=-1)[0].detach().cpu().numpy()
    plt.clf()
    plt.imshow(activation_map.T, origin='lower', aspect='auto', cmap='magma')
    plt.title(f"Alignment Map - Epoch {epoch}")
    plt.xlabel("Time (EEG Frames)")
    plt.ylabel("IndoGPT Tokens")
    plt.yticks(range(len(phrase_tokens)), phrase_tokens)
    plt.colorbar(label="Activation Strength")
    plt.pause(0.01)

# 2. IndoNLG Tokenizer Setup
print("Loading IndoNLGTokenizer for IndoGPT...")
model_name = "indobenchmark/indogpt"
tokenizer = IndoNLGTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size + 1 

# 3. Data Preparation
phrases = ["saya belajar", "makan nasi", "tidur nyenyak", "baca buku"]
encoded = [tokenizer.encode(p) for p in phrases]
shifted = [[t + 1 for t in seq] for seq in encoded]
max_len = max(len(s) for s in shifted)
padded = [s + [0] * (max_len - len(s)) for s in shifted]

target_tokens = torch.tensor(padded, dtype=torch.long).to(device)
tgt_lens = torch.tensor([len(s) for s in shifted], dtype=torch.int32).to(device)

# 4. Model Initialization
from model import ConformerIndoGPTTransducer
eeg_config = {
    "input_dim": 14, "encoder_dim": 144, "decoder_dim": 320, 
    "joint_dim": 320, "vocab_size": vocab_size, "num_layers": 4
}
model = ConformerIndoGPTTransducer(eeg_config).to(device)

# 5. Break the Blank Collapse
with torch.no_grad():
    model.joiner.output_proj.bias[0] = -10.0 

optimizer = optim.Adam(model.parameters(), lr=1e-5)

# --- TRACKING LOSS ---
loss_history = []

# 6. Training Execution
print(f"\n{'Epoch':<8} | {'Loss':<10} | {'U-Dim'}")
print("-" * 50)

plt.ion()
fig1 = plt.figure(figsize=(5, 3)) # Figure for Alignment

total_epoch = 10000

for epoch in range(1, total_epoch+1):
    optimizer.zero_grad()
    
    blank_col = torch.zeros((len(phrases), 1), dtype=torch.long, device=device)
    decoder_input = torch.cat([blank_col, target_tokens], dim=1) 
    
    dummy_eeg = torch.randn(len(phrases), 200, 14).to(device)
    logits = model(dummy_eeg, decoder_input)
    
    loss = torchaudio.functional.rnnt_loss(
        logits=logits,
        targets=target_tokens.to(torch.int32),
        logit_lengths=model.get_encoder_out_lengths(torch.tensor([200]*4)).to(torch.int32).to(device),
        target_lengths=tgt_lens.to(torch.int32),
        blank=0
    )
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Save loss
    loss_history.append(loss.item())
    # Update Alignment Visualization
    plt.figure(fig1.number)
    current_tokens = ["<BLANK>"] + [tokenizer.decode([t]) for t in encoded[0]]
    plot_alignment(logits[0], current_tokens, epoch)
    
    if epoch % 100 == 0:
        print(f"E{epoch:<7} | {loss.item():<10.4f} | {logits.shape[2]}", end="")
        print_gpu_utilization()

plt.ioff()

# --- FINAL LOSS VISUALIZATION ---
fig2 = plt.figure(figsize=(10, 5))
plt.plot(range(1, total_epoch+1), loss_history, marker='o', color='b', linestyle='-')
plt.title("Training Loss Curve (EEG-to-Text)")
plt.xlabel("Epoch")
plt.ylabel("RNN-T Loss")
plt.grid(True)
plt.show() # Shows both the last alignment and the loss curve