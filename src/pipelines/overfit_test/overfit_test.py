"""
Overfit Test (Batch Overfitting Sanity Check)
==============================================

This test verifies that the model can overfit on a very small dataset.
Use 2 random samples from each subject, train and test on the same data.
Should show loss decreasing to near zero and CER close to 0.

Purpose:
- Verify model architecture is correct
- Verify loss computation works
- Verify data pipeline works
- Verify training loop updates weights
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import librosa
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torchaudio.functional as F

warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATASET_CSV = os.path.join(PROJECT_ROOT, 'dataset/cleaned_transcript_mapping.csv')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/overfit_test')

# Model imports
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
from misc.beam_decoder import BeamDecoder
from model import ConformerTransducer

# Training config
CONFIG = {
    'input_dim': 14 * 64,
    'encoder_dim': 256,
    'decoder_dim': 512,
    'joint_dim': 512,
    'vocab_size': None,
    
    # Training - Overfit test: should see loss → 0 quickly
    'batch_size': 1,
    'num_epochs': 20,
    'learning_rate': 1e-3,
    # 'weight_decay': 1e-5,
    'weight_decay': 0,
    
    # Disable dropout for overfit test (want perfect memorization)
    'encoder_dropout': 0.0,
    'decoder_dropout': 0.0,
    
    # Audio processing
    'sample_rate': 256,
    'n_mels': 64,
    'n_fft': 128,      
    'hop_length': 32,
    'f_min': 0.5,
    'f_max': 50.0,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_eeg_channels(eeg_df):
    """Extract 14 EEG channels."""
    eeg_channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                    'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                    'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
    
    if all(ch in eeg_df.columns for ch in eeg_channels):
        return eeg_df[eeg_channels].values
    else:
        raise ValueError("Not all channels found in CSV")

def load_eeg_signal(id_val, subject, gender):
    """Load EEG CSV file for one recording."""
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    
    if not os.path.isdir(csv_folder):
        return None
    
    matching_files = [f for f in os.listdir(csv_folder) 
                      if f.startswith(id_val + '_') and f.endswith('.bp.csv')]
    
    if not matching_files:
        return None
    
    file_path = os.path.join(csv_folder, matching_files[0])
    
    try:
        df = pd.read_csv(file_path, skiprows=1)
        eeg_data = extract_eeg_channels(df)
        return eeg_data
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def compute_log_mel_spectrogram(eeg_signal, config):
    """Compute Log Mel Spectrogram for all EEG channels."""
    n_samples, n_channels = eeg_signal.shape
    mel_specs = []
    
    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float32)
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=config['sample_rate'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_mels=config['n_mels'],
            fmin=config['f_min'],
            fmax=config['f_max']
        )
        mel_spec = np.log(mel_spec + 1e-9)
        mel_specs.append(mel_spec)
    
    mel_spec_stacked = np.stack(mel_specs, axis=0)
    n_time_frames = mel_spec_stacked.shape[2]
    mel_spec_flat = mel_spec_stacked.transpose(2, 0, 1)
    mel_spec_flat = mel_spec_flat.reshape(n_time_frames, -1)
    
    return mel_spec_flat

def load_overfit_dataset(config, samples_per_subject=1):
    """
    Load very small dataset: 2 samples per subject.
    These samples will be used for both training and testing (intentional overfitting).
    """
    print("\n[STEP 1] Load dataset CSV...")
    df = pd.read_csv(DATASET_CSV)
    print(f"Total records available: {len(df)}")
    
    print(f"\n[STEP 2] Select {samples_per_subject} random samples per subject...")
    
    # Group by subject and sample
    np.random.seed(42)
    overfit_data = []
    
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        if len(subject_data) >= samples_per_subject:
            sampled = subject_data.sample(n=samples_per_subject, random_state=42)
            overfit_data.append(sampled)
    
    overfit_df = pd.concat(overfit_data, ignore_index=True).head(1)
    print(f"Selected {len(overfit_df)} samples total")
    print(overfit_df[['id', 'subject', 'sentence']].head(10))
    
    # Load EEG signals and extract features
    print("\n[STEP 3] Load EEG signals and compute Log Mel Spectrograms...")
    
    data = {'features': [], 'targets': [], 'metadata': []}
    
    for idx, row in tqdm(overfit_df.iterrows(), total=len(overfit_df), desc="Processing"):
        id_val = row['id']
        subject = row['subject']
        gender = row['gender']
        sentence = row['sentence']
        
        # Load EEG signal
        eeg_signal = load_eeg_signal(id_val, subject, gender)
        
        if eeg_signal is None:
            print(f"[WARN] Skip {id_val} - signal not found")
            continue
        
        if eeg_signal.shape[0] < config['n_fft']:
            print(f"[WARN] Skip {id_val} - signal too short")
            continue
        
        # Compute features
        mel_spec = compute_log_mel_spectrogram(eeg_signal, config)
        
        data['features'].append(mel_spec)
        data['targets'].append(sentence)
        data['metadata'].append({
            'id': id_val,
            'subject': subject,
            'gender': gender,
            'sentence': sentence
        })
    
    print(f"\n[SUMMARY] Loaded {len(data['features'])} samples")
    
    return data

# ============================================================================
# DATASET CLASS
# ============================================================================

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG-to-Text."""
    
    def __init__(self, features, targets, tokenizer, metadata=None):
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.metadata = metadata or [{}] * len(features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        target_text = self.targets[idx]
        target = torch.LongTensor(self.tokenizer.text_to_int(target_text))
        metadata = self.metadata[idx]
        
        return {
            'feature': feature,
            'target': target,
            'metadata': metadata
        }

def collate_batch(batch):
    """Custom collate for padding sequences."""
    features = [item['feature'] for item in batch]
    targets = [item['target'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    # Pad features
    max_feature_len = max(f.shape[0] for f in features)
    padded_features = []
    feature_lengths = []
    
    for f in features:
        pad_len = max_feature_len - f.shape[0]
        padded = torch.nn.functional.pad(f, (0, 0, 0, pad_len))
        padded_features.append(padded)
        feature_lengths.append(f.shape[0])
    
    features = torch.stack(padded_features)
    feature_lengths = torch.LongTensor(feature_lengths)
    
    # Pad targets
    max_target_len = max(len(t) for t in targets)
    padded_targets = []
    target_lengths = []
    
    for t in targets:
        pad_len = max_target_len - len(t)
        padded = torch.nn.functional.pad(t, (0, pad_len))
        padded_targets.append(padded)
        target_lengths.append(len(t))
    
    targets = torch.stack(padded_targets)
    target_lengths = torch.LongTensor(target_lengths)
    
    return {
        'feature': features,
        'feature_length': feature_lengths,
        'target': targets,
        'target_length': target_lengths,
        'metadata': metadata
    }

# ============================================================================
# CHARACTER ERROR RATE
# ============================================================================

def compute_cer(reference, hypothesis):
    """Compute Character Error Rate using edit distance."""
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
    
    for i in range(len(reference) + 1):
        d[i][0] = i
    for j in range(len(hypothesis) + 1):
        d[0][j] = j
    
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1,
                         d[i][j-1] + 1,
                         d[i-1][j-1] + cost)
    
    return d[len(reference)][len(hypothesis)] / len(reference)

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder=None):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_cer = 0
    num_batches = 0
    count = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        features = batch['feature'].to(device)
        feature_length = batch['feature_length'].to(device)
        targets = batch['target'].to(device)
        target_length = batch['target_length'].to(device)
        
        optimizer.zero_grad()
        
        # Encoder forward
        encoder_out = model.encoder(features)
        
        # Decoder forward with blank token prepended
        batch_size = targets.shape[0]
        blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        decoder_input = torch.cat([blank_col, targets], dim=1)
        hidden_state = model.decoder.init_hidden(batch_size, device)
        decoder_out, _ = model.decoder(decoder_input, hidden_state)
        
        # Joiner forward
        enc_proj = model.joiner.encoder_proj(encoder_out)
        dec_proj = model.joiner.decoder_proj(decoder_out)
        joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
        joint = model.joiner.activation(joint)
        logits = model.joiner.output_proj(joint)
        
        # Compute RNN-T loss
        enc_out_lengths = model.get_encoder_out_lengths(feature_length)
        loss = F.rnnt_loss(
            logits=logits,
            targets=targets.to(torch.int32),
            logit_lengths=enc_out_lengths.to(torch.int32),
            target_lengths=target_length.to(torch.int32),
            blank=0
        )
        
        # Backward & optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute CER using BeamDecoder
        if beam_decoder is not None:
            for i in range(features.shape[0]):
                sample_eeg = features[i:i+1]
                pred_text = beam_decoder.decode(sample_eeg)
                target_text = tokenizer.int_to_text(targets[i].cpu().numpy().tolist())
                cer = compute_cer(target_text, pred_text)
                total_cer += cer
        else:
            # Fallback: skip CER computation if decoder not available
            pass
        
        total_loss += loss.item()
        num_batches += 1
        count += len(targets)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_cer = total_cer / count if count > 0 else 0
    
    return avg_loss, avg_cer

def evaluate(model, test_loader, tokenizer, device, beam_decoder=None):
    """Evaluate model."""
    model.eval()
    total_cer = 0
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            features = batch['feature'].to(device)
            feature_length = batch['feature_length'].to(device)
            targets = batch['target'].to(device)
            target_length = batch['target_length'].to(device)
            
            # Encoder forward
            encoder_out = model.encoder(features)
            
            # Decoder forward
            batch_size = targets.shape[0]
            blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_input = torch.cat([blank_col, targets], dim=1)
            hidden_state = model.decoder.init_hidden(batch_size, device)
            decoder_out, _ = model.decoder(decoder_input, hidden_state)
            
            # Joiner forward
            enc_proj = model.joiner.encoder_proj(encoder_out)
            dec_proj = model.joiner.decoder_proj(decoder_out)
            joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
            joint = model.joiner.activation(joint)
            logits = model.joiner.output_proj(joint)
            
            # Loss
            enc_out_lengths = model.get_encoder_out_lengths(feature_length)
            loss = F.rnnt_loss(
                logits=logits,
                targets=targets.to(torch.int32),
                logit_lengths=enc_out_lengths.to(torch.int32),
                target_lengths=target_length.to(torch.int32),
                blank=0
            )
            total_loss += loss.item()
            
            # CER using BeamDecoder
            if beam_decoder is not None:
                for i in range(features.shape[0]):
                    sample_eeg = features[i:i+1]
                    pred_text = beam_decoder.decode(sample_eeg)
                    target_text = tokenizer.int_to_text(targets[i].cpu().numpy().tolist())
                    cer = compute_cer(target_text, pred_text)
                    total_cer += cer
            else:
                # Fallback: skip CER computation if decoder not available
                pass
            
            count += len(targets)
    
    avg_cer = total_cer / count if count > 0 else 1.0
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    return avg_loss, avg_cer

def train(model, train_loader, test_loader, tokenizer, config, device):
    """Main training loop."""
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    # Create BeamDecoder for CER computation during training
    beam_decoder = BeamDecoder(model, tokenizer, beam_size=3)
    
    history = {'train_loss': [], 'train_cer': [], 'test_loss': [], 'test_cer': []}
    
    print("\n[STEP 4] Training model on small dataset...")
    print("=" * 80)
    print("NOTE: Training and test on SAME data - should see loss → 0 (overfitting)")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        print(f"\n[Epoch {epoch+1}/{config['num_epochs']}]")
        
        # Train
        train_loss, train_cer = train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder)
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        print(f"Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        
        # Test (on same data)
        test_loss, test_cer = evaluate(model, test_loader, tokenizer, device, beam_decoder)
        history['test_loss'].append(test_loss)
        history['test_cer'].append(test_cer)
        print(f"Test Loss:  {test_loss:.4f} | Test CER:  {test_cer:.4f}")
    
    print("\n" + "=" * 80)
    print("EXPECTED RESULT:")
    print("- Train loss should decrease significantly (e.g., 60 → 0.1)")
    print("- Test loss should also decrease (same data)")
    print("- If loss doesn't decrease: Check model, loss, or gradients")
    print("=" * 80)
    
    return history, beam_decoder

def predict_and_save_csv(model, test_loader, tokenizer, data, output_dir, device, beam_decoder=None):
    """
    Make predictions on test data and save to CSV.
    """
    model.eval()
    predictions_list = []
    
    print("\n[STEP 5.5] Generating predictions for CSV...")
    
    metadata_idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            features = batch['feature'].to(device)
            targets = batch['target'].to(device)
            metadata_batch = batch['metadata']
            
            # Get predictions using BeamDecoder
            for i, meta in enumerate(metadata_batch):
                ground_truth = meta['sentence']
                
                # Use BeamDecoder for predictions
                if beam_decoder is not None:
                    sample_eeg = features[i:i+1]
                    pred_text = beam_decoder.decode(sample_eeg)
                else:
                    # Fallback: return empty string if decoder not available
                    pred_text = ""
                
                cer = compute_cer(ground_truth, pred_text)
                
                predictions_list.append({
                    'id': meta['id'],
                    'subject': meta['subject'],
                    'gender': meta['gender'],
                    'sentence': ground_truth,
                    'prediction': pred_text,
                    'cer': cer
                })
    
    # Save to CSV
    predictions_df = pd.DataFrame(predictions_list)
    csv_path = os.path.join(output_dir, 'overfit_predictions.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Predictions saved to {csv_path}")
    
    return predictions_df

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("OVERFIT TEST - Sanity Check")
    print("=" * 80)
    
    # Load small dataset
    data = load_overfit_dataset(CONFIG, samples_per_subject=1)
    
    # Build tokenizer on all data
    print("\n[STEP 3.5] Build Character Tokenizer...")
    all_texts = data['targets']
    tokenizer = CharTokenizer(transcripts=all_texts)
    print(f"Vocab size: {tokenizer.vocab_size()}")
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    
    # Create dataset and dataloader (use same data for train and test)
    print("\n[STEP 3.7] Create PyTorch Datasets and DataLoaders...")
    overfit_dataset = EEGDataset(data['features'], 
                                 data['targets'],
                                 tokenizer,
                                 data['metadata'])
    
    # Use all data for both training and testing
    train_loader = DataLoader(overfit_dataset, 
                             batch_size=CONFIG['batch_size'],
                             collate_fn=collate_batch,
                             shuffle=False)
    test_loader = DataLoader(overfit_dataset, 
                            batch_size=CONFIG['batch_size'],
                            collate_fn=collate_batch,
                            shuffle=False)
    
    print(f"Data: {len(overfit_dataset)} samples")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Build model
    print("\n[STEP 3.9] Build model...")
    model = ConformerTransducer(CONFIG)
    model = model.to(DEVICE)
    print(model)
    
    # Train
    history, beam_decoder = train(model, train_loader, test_loader, tokenizer, CONFIG, DEVICE)
    
    # Generate predictions and save to CSV
    predictions_df = predict_and_save_csv(model, test_loader, tokenizer, data, OUTPUT_DIR, DEVICE, beam_decoder)
    
    # Save results
    print("\n[STEP 5] Save results...")
    history_path = os.path.join(OUTPUT_DIR, 'overfit_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[SAVE] History saved to {history_path}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['test_loss'], label='Test', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Overfit Test - Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # CER
    axes[1].plot(history['train_cer'], label='Train', marker='o')
    axes[1].plot(history['test_cer'], label='Test', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('CER')
    axes[1].set_title('Overfit Test - CER')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'overfit_results.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    print(f"[SAVE] Plot saved to {plot_path}")
    
    print("\n" + "=" * 80)
    print("✓ OVERFIT TEST COMPLETED")
    print("=" * 80)

if __name__ == '__main__':
    main()
