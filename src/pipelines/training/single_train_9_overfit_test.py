"""
OVERFIT TEST PIPELINE (7 SAMPLES)
Untuk debugging Decoding & Tokenizer IndoGPT Transducer
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
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torchaudio.functional as F

from PyEMD import CEEMDAN
from scipy.signal import hilbert
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

import transformers.utils
import transformers.utils.generic

# Monkey Patch
if not hasattr(transformers.utils, 'is_tf_available'):
    transformers.utils.is_tf_available = lambda: False
if not hasattr(transformers.utils.generic, '_is_jax'):
    transformers.utils.generic._is_jax = lambda x: False
if not hasattr(transformers.utils.generic, '_is_tensorflow'):
    transformers.utils.generic._is_tensorflow = lambda x: False
if not hasattr(transformers.utils.generic, '_is_numpy'):
    transformers.utils.generic._is_numpy = lambda x: isinstance(x, np.ndarray)
if not hasattr(transformers.utils.generic, '_is_torch'):
    transformers.utils.generic._is_torch = lambda x: torch.is_tensor(x)
if not hasattr(transformers.utils.generic, '_is_torch_device'):
    transformers.utils.generic._is_torch_device = lambda x: isinstance(x, torch.device)

from indobenchmark import IndoNLGTokenizer
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATASET_CSV = os.path.join(PROJECT_ROOT, 'dataset/cleaned_transcript_mapping.csv')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from model.model import ConformerIndoGPTTransducer
from model.misc.beam_decoder import BeamDecoder

EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

CONFIG = {
    'input_dim': 14 * 65,  
    'encoder_dim': 768,
    'decoder_dim': 768,
    'joint_dim': 768,
    'num_layers': 4,
    'vocab_size': None,
    
    'batch_size': 7,       # Pas 7 sampel
    'num_epochs': 200,     # Cukup untuk overfit
    'learning_rate': 1e-4, 
    'weight_decay': 0.0,   # Matikan weight decay untuk overfit
    
    'encoder_dropout': 0.0, # MATIKAN DROPOUT
    'decoder_dropout': 0.0, # MATIKAN DROPOUT
    
    'sample_rate': 256,
    'hop_length': 8,      
    'win_length': 16,
    'f_min': 0.2,
    'f_max': 45.0,
    
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,  
    
    'num_imfs': 4,         
    'ceemdan_trials': 15,  
    'n_freq_bins': 65,     
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.6):
    frontal_indices = [i for i, ch in enumerate(ch_names) if 'AF3' in ch or 'AF4' in ch]
    if not frontal_indices: return eeg_signal 
    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try: components = ica.fit_transform(eeg_signal) 
    except: return eeg_signal 
    bad_components = []
    for i in range(components.shape[1]):
        is_artifact = False
        for f_idx in frontal_indices:
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                is_artifact = True
                break
        if is_artifact: bad_components.append(i)
    if bad_components: components[:, bad_components] = 0.0
    return ica.inverse_transform(components)

def extract_eeg_channels(eeg_df):
    if all(ch in eeg_df.columns for ch in EEG_CHANNELS): return eeg_df[EEG_CHANNELS].values
    else: raise ValueError("Not all channels found in CSV")

def load_eeg_signal(id_val, subject, gender, config):
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    if not os.path.isdir(csv_folder): return None
    matching_files = [f for f in os.listdir(csv_folder) if f.startswith(id_val + '_') and f.endswith('.bp.csv')]
    if not matching_files: return None
    file_path = os.path.join(csv_folder, matching_files[0])
    try:
        df = pd.read_csv(file_path, skiprows=1)
        signal = extract_eeg_channels(df)
        if config.get('remove_eye_artifacts', True) and signal is not None:
            signal = remove_ocular_artifacts_ica(signal, EEG_CHANNELS, config['ica_threshold'])
        return signal
    except Exception as e:
        return None

def compute_hilbert_spectrum(eeg_signal, config):
    n_samples, n_channels = eeg_signal.shape
    fs, f_min, f_max, n_bins = config['sample_rate'], config['f_min'], config['f_max'], config['n_freq_bins']
    hop_length, win_length, num_imfs = config['hop_length'], config['win_length'], config['num_imfs']
    freq_edges = np.linspace(f_min, f_max, n_bins + 1)
    ceemdan = CEEMDAN(trials=config['ceemdan_trials'], noise_scale=0.2, parallel=False)
    all_channel_spectra = []
    
    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float64)
        imfs = ceemdan(signal, max_imf=num_imfs)
        actual_imfs = imfs.shape[0]
        if actual_imfs > num_imfs: imfs = imfs[:num_imfs, :]
        current_n_samples = n_samples
        hilbert_spec = np.zeros((n_bins, n_samples))
        for i in range(imfs.shape[0]):
            analytic_signal = hilbert(imfs[i])
            amp, phase = np.abs(analytic_signal), np.unwrap(np.angle(analytic_signal))
            freq = (np.diff(phase) / (2.0*np.pi) * fs)
            freq = np.insert(freq, 0, freq[0])
            bin_indices = np.digitize(freq, freq_edges) - 1
            for t in range(n_samples):
                b = bin_indices[t]
                if 0 <= b < n_bins: hilbert_spec[b, t] += (amp[t] ** 2) 
        if current_n_samples > win_length:
            remainder = (current_n_samples - win_length) % hop_length
            if remainder > 0:
                pad_length = hop_length - remainder
                hilbert_spec = np.pad(hilbert_spec, ((0, 0), (0, pad_length)), mode='constant')
                current_n_samples += pad_length
        if current_n_samples < win_length:
            n_frames = 0
            framed_spec = np.zeros((n_bins, 0)) 
        else:
            n_frames = 1 + (current_n_samples - win_length) // hop_length
            framed_spec = np.zeros((n_bins, n_frames))
            for t_idx in range(n_frames):
                start = t_idx * hop_length
                end = start + win_length  
                framed_spec[:, t_idx] = np.mean(hilbert_spec[:, start:end], axis=1)  
        all_channel_spectra.append(framed_spec)
        
    all_channel_spectra = np.array(all_channel_spectra)
    features_transposed = all_channel_spectra.transpose(2, 0, 1)
    features_flat = features_transposed.reshape(features_transposed.shape[0], -1)
    features_flat = np.log(features_flat + 1e-9)
    features_flat = (features_flat - np.mean(features_flat, axis=0)) / (np.std(features_flat, axis=0) + 1e-6)
    return features_flat.astype(np.float32)

def load_overfit_dataset(config, target_subject, num_samples=7):
    print(f"\n[STEP 1] Load EXACTLY {num_samples} samples for OVERFIT TEST...")
    df = pd.read_csv(DATASET_CSV)
    df = df[df['subject'] == target_subject].copy()
    
    features, targets, metadata = [], [], []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Samples"):
        id_val, subject, gender, sentence = row['id'], row['subject'], row['gender'], row['sentence']
        eeg_signal = load_eeg_signal(id_val, subject, gender, config)
        
        if eeg_signal is None or eeg_signal.shape[0] < config['hop_length']:
            continue
            
        hilbert_features = compute_hilbert_spectrum(eeg_signal, config)
        features.append(hilbert_features)
        targets.append(sentence)
        metadata.append({'id': id_val, 'subject': subject, 'gender': gender, 'sentence': sentence})
        
        if len(features) == num_samples:
            break
            
    # Copy data to train, val, and test to force overfit evaluation
    data = {
        'train': {'features': features, 'targets': targets, 'metadata': metadata},
        'val': {'features': features, 'targets': targets, 'metadata': metadata},
        'test': {'features': features, 'targets': targets, 'metadata': metadata}
    }
    print(f"[SUMMARY] Locked {len(features)} samples for pure memorization test.")
    return data

class EEGDataset(Dataset):
    def __init__(self, features, targets, tokenizer, metadata=None):
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.metadata = metadata or [{}] * len(features)
    
    def __len__(self): return len(self.features)
    
    def __getitem__(self, idx):
        encoded_tokens = self.tokenizer.encode(self.targets[idx])
        shifted_tokens = [t + 1 for t in encoded_tokens]
        return {
            'feature': torch.FloatTensor(self.features[idx]),
            'target': torch.LongTensor(shifted_tokens),
            'metadata': self.metadata[idx]
        }

def collate_batch(batch):
    features = [item['feature'] for item in batch]
    targets = [item['target'] for item in batch]
    max_feature_len = max(f.shape[0] for f in features)
    padded_features = [torch.nn.functional.pad(f, (0, 0, 0, max_feature_len - f.shape[0])) for f in features]
    max_target_len = max(len(t) for t in targets)
    padded_targets = [torch.nn.functional.pad(t, (0, max_target_len - len(t))) for t in targets]
    return {
        'feature': torch.stack(padded_features),
        'feature_length': torch.LongTensor([f.shape[0] for f in features]),
        'target': torch.stack(padded_targets),
        'target_length': torch.LongTensor([len(t) for t in targets]),
        'metadata': [item['metadata'] for item in batch]
    }

def compute_cer(reference, hypothesis):
    if len(reference) == 0: return 1.0 if len(hypothesis) > 0 else 0.0
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1): d[i][0] = i
    for j in range(len(hypothesis) + 1): d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(reference)][len(hypothesis)] / len(reference)

def train_epoch(model, train_loader, optimizer, tokenizer, device):
    total_loss = 0
    model.train()
    for batch in train_loader:
        features, feature_length = batch['feature'].to(device), batch['feature_length'].to(device) 
        targets, target_length = batch['target'].to(device), batch['target_length'].to(device) 
        
        optimizer.zero_grad()
        batch_size = targets.shape[0]
        blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        decoder_input = torch.cat([blank_col, targets], dim=1) 
        
        logits = model(features, decoder_input)
        enc_out_lengths = model.get_encoder_out_lengths(feature_length)
        
        loss = F.rnnt_loss(
            logits=logits, targets=targets.to(torch.int32),
            logit_lengths=enc_out_lengths.to(torch.int32),
            target_lengths=target_length.to(torch.int32), blank=0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, loader, tokenizer, device, beam_decoder, epoch):
    model.eval()
    total_cer = 0
    count = 0
    
    # KITA INTIP APA YANG DIHASILKAN MODEL
    print(f"\n--- HASIL DECODING EPOCH {epoch} ---")
    with torch.no_grad():
        for batch in loader:
            features = batch['feature'].to(device)
            targets = batch['target'].to(device)
            
            for i in range(features.shape[0]):
                pred_text = beam_decoder.decode(features[i:i+1])
                
                unshifted_target = [t.item() - 1 for t in targets[i] if t.item() > 0]
                target_text = tokenizer.decode(unshifted_target)
                
                total_cer += compute_cer(target_text, pred_text)
                count += 1
                
                # Print 3 sampel pertama untuk dilihat
                if i < 3:
                    print(f"Target : '{target_text}'")
                    print(f"Predict: '{pred_text}'\n")
                    
    return total_cer / count

def main():
    TARGET_SUBJECT = 'SUB1' 
    print("=" * 80)
    print("OVERFIT TEST: Memaksa Model Menghafal 7 Sampel")
    print("=" * 80)
    
    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indogpt")
    def dummy_pad(encoded_inputs, **kwargs): return encoded_inputs
    tokenizer.pad = dummy_pad
    if not hasattr(tokenizer, 'int_to_text'): tokenizer.int_to_text = tokenizer.decode
        
    CONFIG['vocab_size'] = tokenizer.vocab_size + 1
    
    data = load_overfit_dataset(CONFIG, TARGET_SUBJECT, num_samples=7)
    
    train_dataset = EEGDataset(data['train']['features'], data['train']['targets'], tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_batch)
    
    model = ConformerIndoGPTTransducer(CONFIG).to(DEVICE)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=CONFIG['learning_rate'])
    beam_decoder = BeamDecoder(model, tokenizer, beam_size=3, blank_id=0)
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        loss = train_epoch(model, train_loader, optimizer, tokenizer, DEVICE)
        
        # Validasi/Print log setiap 10 epoch agar layar tidak penuh
        if epoch % 10 == 0 or epoch == 1:
            cer = evaluate(model, train_loader, tokenizer, DEVICE, beam_decoder, epoch)
            print(f"[Epoch {epoch}] Loss: {loss:.4f} | Train CER: {cer:.4f}")

if __name__ == '__main__':
    main()