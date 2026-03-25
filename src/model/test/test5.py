import torch
import os
import sys

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# Anchor to the Root directory
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))

# Add /src to sys.path to access the model class
src_path = os.path.join(root_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from model.model import ConformerIndoGPTTransducer
from model.misc.beam_decoder import BeamDecoder
from indobenchmark import IndoNLGTokenizer

# --- 2. CONFIGURATION ---
exp_name = "exp_name_example"  # Change this to match your experiment folder
results_path = os.path.join(root_dir, "experiments", exp_name, "results", "model")
weights_path = os.path.join(results_path, "model.pt")
tokenizer_path = os.path.join(results_path, "tokenizer")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_inference():
    # --- 3. LOAD TOKENIZER ---
    print(f"Loading tokenizer from: {tokenizer_path}")
    # This will look for vocab/sentencepiece files in your results folder
    tokenizer = IndoNLGTokenizer.from_pretrained(tokenizer_path)
    vocab_size = tokenizer.vocab_size + 1

    # --- 4. INITIALIZE MODEL ARCHITECTURE ---
    # Note: Architecture must match the training config exactly
    eeg_config = {
        "input_dim": 14, 
        "encoder_dim": 144, 
        "decoder_dim": 320, 
        "joint_dim": 320, 
        "vocab_size": vocab_size, 
        "num_layers": 4
    }
    model = ConformerIndoGPTTransducer(eeg_config).to(device)

    # --- 5. LOAD WEIGHTS ---
    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode

    eeg_debug_path = os.path.join(results_path, "fixed_eeg_debug.pt")
    
    if os.path.exists(eeg_debug_path):
        print(f"Loading debug EEG data from: {eeg_debug_path}")
        # Load and move to the current device (CUDA)
        test_eeg_batch = torch.load(eeg_debug_path, map_location=device)
    else:
        print("Debug EEG not found. Generating fresh noise...")
        test_eeg_batch = torch.randn(2, 200, 14).to(device)

    # --- 6. PERFORM DECODING ---
    # We use Beam Search as discussed yesterday
    beam_decoder = BeamDecoder(model, tokenizer, beam_size=5)
    
    print("\n--- Running Overfitting Check ---")
    # test_eeg_batch has shape (2, 200, 14)
    for i in range(test_eeg_batch.size(0)):
        sample_eeg = test_eeg_batch[i:i+1] # Slice to keep 3D shape (1, 200, 14)
        result = beam_decoder.decode(sample_eeg)
        print(f"Sample {i+1} Result: '{result}'")
    print("Inference successful.")

if __name__ == "__main__":
    run_inference()