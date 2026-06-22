import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import re
from collections import defaultdict, Counter
from tqdm import tqdm
from pathlib import Path
import librosa
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import torchaudio.functional as F

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATASET_CSV = os.path.join(PROJECT_ROOT, 'dataset/cleaned_transcript_mapping.csv')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
from misc.beam_decoder import BeamDecoder
from model import ConformerTransducer

CONFIG = {
    'input_dim': 14 * 80,  # 14 channels x 80 mel frequency bins
    'encoder_dim': 256,
    'decoder_dim': 512,
    'joint_dim': 512,
    'vocab_size': None,  # Akan diupdate setelah tokenizer dibuat
    
    'batch_size': 8,
    'num_epochs': 5,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    
    'sample_rate': 256,  # EEG sampling rate
    'n_mels': 80,
    'n_fft': 32,          # ~125ms window at 256 Hz
    'hop_length': 8,      # ~31ms stride at 256 Hz
    'f_min': 0.5,
    'f_max': 50.0,  # Bandpass sudah dilakukan
    
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

def extract_eeg_channels(eeg_df):
    """
    Ekstrak 14 channel EEG dari dataframe.
    Channel: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    """
    eeg_channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                    'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                    'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
    
    if all(ch in eeg_df.columns for ch in eeg_channels):
        return eeg_df[eeg_channels].values
    else:
        raise ValueError(f"Tidak semua channel ditemukan di CSV")

def load_eeg_signal(id_val, subject, gender):
    """
    Load EEG CSV file untuk satu recording.
    Return: numpy array dengan shape (n_samples, 14 channels)
    """
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    
    if not os.path.isdir(csv_folder):
        return None
    
    # Cari file dengan prefix ID dan suffix .bp.csv
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
        print(f"[ERROR] Gagal load {file_path}: {e}")
        return None

def compute_log_mel_spectrogram(eeg_signal, config):
    """
    Hitung Log Mel Spectrogram untuk seluruh channel EEG.
    
    Input:
        eeg_signal: numpy array shape (n_samples, n_channels)
        config: dictionary dengan parameter audio
    
    Output:
        mel_spec: numpy array shape (n_channels, n_mel_bins, n_time_frames)
    """
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
    
    # Stack: (n_channels, n_mels, n_time_frames)
    mel_spec_stacked = np.stack(mel_specs, axis=0)
    
    # Reshape to (n_time_frames, n_channels * n_mels) seperti audio features
    n_time_frames = mel_spec_stacked.shape[2]
    mel_spec_flat = mel_spec_stacked.transpose(2, 0, 1)  # (n_time, n_ch, n_mels)
    mel_spec_flat = mel_spec_flat.reshape(n_time_frames, -1)  # (n_time, n_ch*n_mels)
    
    return mel_spec_flat

def split_dataset_by_sentence(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Split dataset dengan constraint: kalimat yang sama harus ada di bagian yang sama.
    """
    np.random.seed(seed)
    
    unique_sentences = df['sentence'].unique()
    n_unique = len(unique_sentences)
    
    train_count = int(n_unique * train_ratio)
    val_count = int(n_unique * val_ratio)
    
    shuffled_sentences = np.random.permutation(unique_sentences)
    
    train_sentences = set(shuffled_sentences[:train_count])
    val_sentences = set(shuffled_sentences[train_count:train_count+val_count])
    test_sentences = set(shuffled_sentences[train_count+val_count:])
    
    df['split'] = df['sentence'].apply(
        lambda x: 'train' if x in train_sentences 
                  else ('val' if x in val_sentences else 'test')
    )
    
    return df

def load_and_preprocess_dataset(config):
    """
    Load dataset CSV, preprocess EEG signals, dan compute features.
    Return: dictionary dengan train/val/test features dan targets
    """
    print("\n[STEP 1] Load dataset CSV...")
    df = pd.read_csv(DATASET_CSV)
    print(f"Total records: {len(df)}")
    
    print("[STEP 2] Split dataset (70% train, 10% val, 20% test)...")
    df = split_dataset_by_sentence(df, 
                                   train_ratio=0.7, 
                                   val_ratio=0.1, 
                                   test_ratio=0.2)
    
    print(df['split'].value_counts())
    
    print("\n[STEP 3] Load & process EEG signals, compute Log Mel Spectrograms...")
    
    data = {'train': {'features': [], 'targets': [], 'metadata': []},
            'val': {'features': [], 'targets': [], 'metadata': []},
            'test': {'features': [], 'targets': [], 'metadata': []}}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        id_val = row['id']
        subject = row['subject']
        gender = row['gender']
        sentence = row['sentence']
        split = row['split']
        
        eeg_signal = load_eeg_signal(id_val, subject, gender)
        
        if eeg_signal is None:
            print(f"[WARN] Skip {id_val} - signal not found")
            continue
        
        if eeg_signal.shape[0] < config['n_fft']:
            print(f"[WARN] Skip {id_val} - signal too short")
            continue
        
        mel_spec = compute_log_mel_spectrogram(eeg_signal, config)
        
        data[split]['features'].append(mel_spec)
        data[split]['targets'].append(sentence)
        data[split]['metadata'].append({
            'id': id_val,
            'subject': subject,
            'gender': gender,
            'sentence': sentence
        })
    
    print(f"\n[SUMMARY] Loaded {len(data['train']['features'])} train, "
          f"{len(data['val']['features'])} val, "
          f"{len(data['test']['features'])} test samples")
    
    return data

# from /src/model/model.py which uses Encoder, Decoder, Joiner, and misc modules

class EEGDataset(Dataset):
    """PyTorch Dataset untuk EEG-to-Text."""
    
    def __init__(self, features, targets, tokenizer, metadata=None):
        """
        Args:
            features: list of numpy arrays (time_steps, n_features)
            targets: list of strings
            tokenizer: CharTokenizer instance
            metadata: list of dicts dengan info recording
        """
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
    """Custom collate untuk padding sequences."""
    features = [item['feature'] for item in batch]
    targets = [item['target'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
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

def compute_cer(reference, hypothesis):
    """
    Compute Character Error Rate menggunakan edit distance.
    """
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
            d[i][j] = min(d[i-1][j] + 1,      # deletion
                         d[i][j-1] + 1,       # insertion
                         d[i-1][j-1] + cost)  # substitution
    
    return d[len(reference)][len(hypothesis)] / len(reference)

def train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder=None):
    """Train satu epoch menggunakan RNN-T loss dan compute CER."""
    model.train()
    total_loss = 0
    total_cer = 0
    num_batches = 0
    count = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        features = batch['feature'].to(device)  # (batch, time, features)
        feature_length = batch['feature_length'].to(device)  # (batch,)
        targets = batch['target'].to(device)     # (batch, target_len)
        target_length = batch['target_length'].to(device)  # (batch,)
        
        optimizer.zero_grad()
        
        encoder_out = model.encoder(features)  # (batch, enc_time, encoder_dim)
        
        batch_size = targets.shape[0]
        blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        decoder_input = torch.cat([blank_col, targets], dim=1)  # (batch, target_len+1)
        hidden_state = model.decoder.init_hidden(batch_size, device)
        decoder_out, _ = model.decoder(decoder_input, hidden_state)  # (batch, target_len+1, decoder_dim)
        
        # encoder_out: (batch, enc_time, encoder_dim)
        # decoder_out: (batch, dec_time, decoder_dim)
        enc_proj = model.joiner.encoder_proj(encoder_out)  # (batch, enc_time, joint_dim)
        dec_proj = model.joiner.decoder_proj(decoder_out)  # (batch, dec_time, joint_dim)
        
        # (batch, enc_time, 1, joint_dim) + (batch, 1, dec_time, joint_dim)
        joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)  # (batch, enc_time, dec_time, joint_dim)
        joint = model.joiner.activation(joint)
        logits = model.joiner.output_proj(joint)  # (batch, enc_time, dec_time, vocab_size)
        
        # IMPORTANT: Use encoder output lengths (after subsampling), not input feature lengths
        enc_out_lengths = model.get_encoder_out_lengths(feature_length)
        loss = F.rnnt_loss(
            logits=logits,
            targets=targets.to(torch.int32),
            logit_lengths=enc_out_lengths.to(torch.int32),
            target_lengths=target_length.to(torch.int32),
            blank=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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

def evaluate(model, val_loader, tokenizer, device, beam_decoder=None):
    """Evaluate model pada validation set menggunakan RNN-T loss."""
    model.eval()
    total_cer = 0
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            features = batch['feature'].to(device)  # (batch, time, features)
            feature_length = batch['feature_length'].to(device)
            targets = batch['target'].to(device)     # (batch, target_len)
            target_length = batch['target_length'].to(device)
            
            encoder_out = model.encoder(features)
            
            batch_size = targets.shape[0]
            blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_input = torch.cat([blank_col, targets], dim=1)
            hidden_state = model.decoder.init_hidden(batch_size, device)
            decoder_out, _ = model.decoder(decoder_input, hidden_state)
            
            enc_proj = model.joiner.encoder_proj(encoder_out)
            dec_proj = model.joiner.decoder_proj(decoder_out)
            joint = enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1)
            joint = model.joiner.activation(joint)
            logits = model.joiner.output_proj(joint)  # (batch, enc_time, dec_time, vocab_size)
            
            # RNN-T Loss using torchaudio.functional
            # IMPORTANT: Use encoder output lengths (after subsampling), not input feature lengths
            enc_out_lengths = model.get_encoder_out_lengths(feature_length)
            loss = F.rnnt_loss(
                logits=logits,
                targets=targets.to(torch.int32),
                logit_lengths=enc_out_lengths.to(torch.int32),
                target_lengths=target_length.to(torch.int32),
                blank=0
            )
            total_loss += loss.item()
            
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
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    return avg_loss, avg_cer

def train(model, train_loader, val_loader, tokenizer, config, device):
    """Main training loop."""
    optimizer = optim.Adam(model.parameters(), 
                          lr=config['learning_rate'],
                          weight_decay=config['weight_decay'])
    
    beam_decoder = BeamDecoder(model, tokenizer, beam_size=3)
    
    history = {'train_loss': [], 'train_cer': [], 'val_loss': [], 'val_cer': []}
    best_cer = float('inf')
    best_model = None
    
    print("\n[STEP 6] Training model...")
    print("=" * 80)
    
    for epoch in range(config['num_epochs']):
        print(f"\n[Epoch {epoch+1}/{config['num_epochs']}]")
        
        train_loss, train_cer = train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder)
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        print(f"Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        
        val_loss, val_cer = evaluate(model, val_loader, tokenizer, device, beam_decoder)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        print(f"Val Loss: {val_loss:.4f} | Val CER: {val_cer:.4f}")
        
        if val_cer < best_cer:
            best_cer = val_cer
            best_model = model.state_dict()
            print("[SAVE] Best model saved")
    
    if best_model is not None:
        model.load_state_dict(best_model)
    
    print("\n" + "=" * 80)
    return history

def predict(model, test_loader, tokenizer, device, max_steps=500):
    """
    Predict pada test set menggunakan greedy RNN-T decoding.
    """
    model.eval()
    predictions = []
    
    print("\n[STEP 7+8] Predicting on test set & computing CER...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            features = batch['feature'].to(device)  # (1, time, features) untuk batch_size=1
            metadata = batch['metadata']
            
            encoder_out = model.encoder(features)  # (1, enc_time, encoder_dim)
            enc_time = encoder_out.shape[1]
            
            pred_tokens = []
            
            for enc_step in range(enc_time):

                enc_current = encoder_out[:, enc_step:enc_step+1, :]  # (1, 1, encoder_dim)
                
                decoder_input_ids = [tokenizer.blank_id] + pred_tokens
                decoder_input = torch.LongTensor([decoder_input_ids]).to(device)  # (1, dec_len)
                
                decoder_out, _ = model.decoder(decoder_input, None)  # (1, dec_len, decoder_dim)
                
                decoder_last = decoder_out[:, -1:, :]  # (1, 1, decoder_dim)
                
                # Joiner: compute logits untuk current encoder step
                enc_proj = model.joiner.encoder_proj(enc_current)  # (1, 1, joint_dim)
                dec_proj = model.joiner.decoder_proj(decoder_last)  # (1, 1, joint_dim)
                
                joint = enc_proj + dec_proj  # (1, 1, joint_dim)
                joint = model.joiner.activation(joint)
                logits = model.joiner.output_proj(joint)  # (1, 1, vocab_size)
                
                # Greedy: pick token dengan probability tertinggi
                next_token = torch.argmax(logits[0, 0, :]).item()
                
                # If not blank token, add to prediction
                if next_token != tokenizer.blank_id:
                    pred_tokens.append(next_token)
            
            pred_text = tokenizer.int_to_text(pred_tokens)
            target_text = metadata[0]['sentence']
            cer = compute_cer(target_text, pred_text)
            
            predictions.append({
                'id': metadata[0].get('id', ''),
                'subject': metadata[0].get('subject', ''),
                'gender': metadata[0].get('gender', ''),
                'target_sentence': target_text,
                'predicted_sentence': pred_text,
                'cer': cer
            })
    
    return predictions

def plot_training_history(history, output_dir):
    """
    Plot training loss, validation metrics, dan CER history.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, marker='s', markersize=4, label='Val Loss')
    axes[0, 0].fill_between(epochs, history['train_loss'], history['val_loss'], alpha=0.2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Loss', fontsize=11)
    axes[0, 0].set_title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, history['train_cer'], 'g-', linewidth=2, marker='^', markersize=4, label='Train CER')
    axes[0, 1].plot(epochs, history['val_cer'], 'm-', linewidth=2, marker='v', markersize=4, label='Val CER')
    axes[0, 1].fill_between(epochs, history['train_cer'], history['val_cer'], alpha=0.2, color='cyan')
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Character Error Rate', fontsize=11)
    axes[0, 1].set_title('Training vs Validation CER', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].semilogy(epochs, history['train_loss'], 'b-', linewidth=2, marker='o', label='Train Loss')
    axes[1, 0].semilogy(epochs, history['val_loss'], 'r-', linewidth=2, marker='s', label='Val Loss')
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Loss (log scale)', fontsize=11)
    axes[1, 0].set_title('Loss Convergence (Log Scale)', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, which='both')
    
    train_cer_improvement = 100 * (history['train_cer'][0] - history['train_cer'][-1]) / (history['train_cer'][0] + 1e-9)
    val_cer_improvement = 100 * (history['val_cer'][0] - history['val_cer'][-1]) / (history['val_cer'][0] + 1e-9)
    
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.9, 'Training Summary', ha='center', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    
    summary_text = f"""
Train Loss:  {history['train_loss'][0]:.4f} → {history['train_loss'][-1]:.4f}
Val Loss:    {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}

Train CER:   {history['train_cer'][0]:.4f} → {history['train_cer'][-1]:.4f}
             (↓ {train_cer_improvement:.1f}%)

Val CER:     {history['val_cer'][0]:.4f} → {history['val_cer'][-1]:.4f}
             (↓ {val_cer_improvement:.1f}%)

Best Val CER: {min(history['val_cer']):.4f} (Epoch {history['val_cer'].index(min(history['val_cer'])) + 1})
    """
    
    ax4.text(0.1, 0.75, summary_text, ha='left', va='top', fontsize=10, 
            family='monospace', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.axis('off')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"[SAVE] Training history plot saved to {plot_path}")
    plt.close()

def main():
    print("=" * 80)
    print("EEG-to-Text Conformer-Transducer Training Pipeline")
    print("=" * 80)
    
    data = load_and_preprocess_dataset(CONFIG)
    
    print("\n[STEP 4] Build Character Tokenizer...")
    # Use ALL texts (train+val+test) so all characters are captured
    all_texts = data['train']['targets'] + data['val']['targets'] + data['test']['targets']
    tokenizer = CharTokenizer(transcripts=all_texts)
    print(f"Vocab size: {tokenizer.vocab_size()}")
    
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    
    tokenizer_path = os.path.join(OUTPUT_DIR, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    print(f"[SAVE] Tokenizer saved to {tokenizer_path}")
    
    print("\n[STEP 5] Create PyTorch Datasets...")
    train_dataset = EEGDataset(data['train']['features'], 
                               data['train']['targets'],
                               tokenizer,
                               data['train']['metadata'])
    val_dataset = EEGDataset(data['val']['features'],
                            data['val']['targets'],
                            tokenizer,
                            data['val']['metadata'])
    test_dataset = EEGDataset(data['test']['features'],
                             data['test']['targets'],
                             tokenizer,
                             data['test']['metadata'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                           shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=1,
                            shuffle=False, collate_fn=collate_batch)
    
    print("\n[STEP 5b] Build model...")
    model = ConformerTransducer(CONFIG)
    model.tokenizer = tokenizer
    model = model.to(DEVICE)
    print(model)
    
    history = train(model, train_loader, val_loader, tokenizer, CONFIG, DEVICE)
    
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[SAVE] Training history saved to {history_path}")
    
    plot_training_history(history, OUTPUT_DIR)
    
    model_path = os.path.join(OUTPUT_DIR, 'model.pt')
    torch.save(model.state_dict(), model_path)
    print(f"[SAVE] Model saved to {model_path}")
    
    predictions = predict(model, test_loader, tokenizer, DEVICE)
    
    print("\n[STEP 9] Save results...")
    if predictions:
        results_df = pd.DataFrame(predictions)
        results_csv = os.path.join(OUTPUT_DIR, 'test_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"[SAVE] Results saved to {results_csv}")
        
        avg_test_cer = results_df['cer'].mean()
        print(f"[SUMMARY] Average Test CER: {avg_test_cer:.4f}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
