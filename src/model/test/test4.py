import torch
import torchaudio.functional as F
from indobenchmark import IndoNLGTokenizer
import os
import sys

# --- 1. PATH DYNAMICS ---
# This anchors everything to the Root directory regardless of where you launch from
# current_dir: /src/model/test
current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir: / (The root of your GitRepo)
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))

# Add /src to sys.path so we can import 'model.model' (your base mentahan)
src_path = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# Import your model logic from /src/model/model.py
from model.model import ConformerIndoGPTTransducer
from model.misc.beam_decoder import BeamDecoder

# --- 2. EXPERIMENT DIRECTORY SETUP ---
exp_name = "exp_name_example"  # Define your experiment name here
results_base = os.path.join(root_dir, "experiments", exp_name, "results")
model_save_dir = os.path.join(results_base, "model")
tokenizer_save_dir = os.path.join(model_save_dir, "tokenizer")

# Create the full path structure: /experiments/exp_name/results/model/tokenizer
os.makedirs(tokenizer_save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. INITIALIZATION ---
print(f"Loading Tokenizer & Model for Experiment: {exp_name}")
tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indogpt")
vocab_size = tokenizer.vocab_size + 1

eeg_config = {
    "input_dim": 14, "encoder_dim": 144, "decoder_dim": 320, 
    "joint_dim": 320, "vocab_size": vocab_size, "num_layers": 4
}
model = ConformerIndoGPTTransducer(eeg_config).to(device)

# --- 4. TRAINING SIMULATION ---
# (Using fixed data for the memorization check we discussed)
phrases = ["saya belajar", "makan nasi"]
encoded = [tokenizer.encode(p) for p in phrases]
shifted = [[t + 1 for t in seq] for seq in encoded]
actual_max_len = max(len(s) for s in shifted)
padded_shifted = [s + [0] * (actual_max_len - len(s)) for s in shifted]

eeg_save_path = os.path.join(model_save_dir, "fixed_eeg_debug.pt")

target_tokens = torch.tensor(padded_shifted).to(device)
tgt_lens = torch.tensor([len(s) for s in shifted], dtype=torch.int32).to(device)
fixed_eeg = torch.randn(2, 200, 14).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("Starting training...")
total_epoch = 5000
for epoch in range(1, total_epoch+1):
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

# --- 5. SAVING MECHANISM ---
print(f"\nTraining finished. Saving to: {model_save_dir}")

# 1. Save Model Weights (PyTorch standard)
torch.save(model.state_dict(), os.path.join(model_save_dir, "model.pt"))

# 2. Save the Fixed EEG tensor for inference testing
torch.save(fixed_eeg.cpu(), eeg_save_path) # Move to CPU before saving for compatibility

# 2. Save Tokenizer Manually
import shutil

try:
    # Get the directory where the tokenizer was downloaded (Hugging Face Cache)
    # We use the internal properties of the tokenizer to find its source
    cache_dir = tokenizer.vocab_file if hasattr(tokenizer, 'vocab_file') else None
    
    if cache_dir and os.path.exists(cache_dir):
        # The vocab_file is usually a path to 'sentencepiece.bpe.model' or similar
        source_dir = os.path.dirname(cache_dir)
        
        # Files we want to copy
        files_to_copy = [
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "sentencepiece.bpe.model", # IndoGPT uses this
            "added_tokens.json"
        ]
        
        for filename in files_to_copy:
            src_file = os.path.join(source_dir, filename)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(tokenizer_save_dir, filename))
        
        print(f"Successfully copied tokenizer files to {tokenizer_save_dir}")
    else:
        # Fallback: Just save the config and we'll download vocab on load if needed
        # We use the basic dict saving to avoid the NotImplementedError
        import json
        with open(os.path.join(tokenizer_save_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer.init_kwargs, f)
        print("Saved tokenizer configuration JSON as fallback.")

except Exception as e:
    print(f"Manual copy failed: {e}")