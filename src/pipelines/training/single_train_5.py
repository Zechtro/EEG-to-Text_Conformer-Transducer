"""
Full Training Pipeline for EEG-to-Text Conformer-Transducer (Single Subject)
============================================================================

Fitur:
1. Filter dataset berdasarkan satu subjek spesifik
2. Split dataset berdasarkan kalimat (70% Train, 10% Val, 20% Test)
3. Ekstraksi fitur menggunakan Hilbert Spectrum (CEEMDAN + HHT Binning 14x65)
4. Training menggunakan RNN-T loss dengan evaluasi CER (BeamDecoder)
5. Menyimpan model terbaik dan hasil dengan penamaan dinamis per subjek
6. Bebas spam log multiprocessing
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

# Import library untuk Hilbert Spectrum
from PyEMD import CEEMDAN
from scipy.signal import hilbert
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI PATH & PARAMETER
# ============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATASET_CSV = os.path.join(PROJECT_ROOT, 'dataset/cleaned_transcript_mapping.csv')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

# Definisi Channel Global agar bisa dibaca oleh ICA
EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

CONFIG = {
    'input_dim': 14 * 65,  # 14 channel * 65 pita frekuensi (Sama seperti STFT)
    'encoder_dim': 128,
    'decoder_dim': 128,
    'joint_dim': 128,
    'vocab_size': None,
    
    'batch_size': 7,
    'num_epochs': 200, 
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,  # Ditingkatkan untuk mengurangi overfit
    
    'encoder_dropout': 0.2, # Ditingkatkan untuk mengurangi overfit
    'decoder_dropout': 0.2, # Ditingkatkan untuk mengurangi overfit
    
    'sample_rate': 256,
    'hop_length': 16,      # Ukuran window downsampling waktu
    'win_length': 32,
    # 'hop_length': 8,      # Ukuran window downsampling waktu
    # 'win_length': 16,
    'f_min': 0.2,
    'f_max': 45.0,
    
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,  # Jika korelasi komponen dgn AF3/AF4 > 0.6, anggap sbg kedipan
    
    # Parameter CEEMDAN & Hilbert Spectrum
    'num_imfs': 4,         # Jumlah IMF yang diekstrak per channel
    'ceemdan_trials': 15,  # Jumlah ensemble trial CEEMDAN
    'n_freq_bins': 65,     # Resolusi pita frekuensi
    
    'train_ratio': 0.7,
    'val_ratio': 0.1,
    'test_ratio': 0.2,
}

# Inisialisasi Device HANYA sebagai variabel (Print dipindah ke fungsi main)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# UTILITY FUNCTIONS & FEATURE EXTRACTION (HILBERT SPECTRUM)
# ============================================================================

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.6):
    """
    Menghilangkan artefak mata (kedipan/gerakan) secara otomatis menggunakan FastICA.
    """
    # 1. Cari indeks channel frontal (AF3 dan AF4 berada tepat di atas mata)
    frontal_indices = [i for i, ch in enumerate(ch_names) if 'AF3' in ch or 'AF4' in ch]
    
    if not frontal_indices:
        return eeg_signal # Fallback jika channel frontal tidak ada

    # 2. Dekomposisi sinyal menggunakan FastICA
    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try:
        components = ica.fit_transform(eeg_signal) # Shape: (n_samples, n_components)
    except:
        return eeg_signal # Fallback (Lewati) jika ICA gagal konvergen

    # 3. Identifikasi komponen buruk (Yang sangat mirip dengan sinyal di AF3/AF4)
    bad_components = []
    for i in range(components.shape[1]):
        is_artifact = False
        for f_idx in frontal_indices:
            # Hitung korelasi Pearson
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                is_artifact = True
                break
        if is_artifact:
            bad_components.append(i)

    # 4. Nol-kan (Hapus) komponen yang teridentifikasi sebagai artefak mata
    if bad_components:
        components[:, bad_components] = 0.0

    # 5. Rakit kembali sinyal otak yang sudah bersih
    cleaned_signal = ica.inverse_transform(components)
    return cleaned_signal

def extract_eeg_channels(eeg_df):
    eeg_channels = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                    'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                    'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']
    if all(ch in eeg_df.columns for ch in eeg_channels):
        return eeg_df[eeg_channels].values
    else:
        raise ValueError("Not all channels found in CSV")

def load_eeg_signal(id_val, subject, gender, config):
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    if not os.path.isdir(csv_folder): return None
    
    matching_files = [f for f in os.listdir(csv_folder) 
                      if f.startswith(id_val + '_') and f.endswith('.bp.csv')]
    if not matching_files: return None
    
    file_path = os.path.join(csv_folder, matching_files[0])
    try:
        df = pd.read_csv(file_path, skiprows=1)
        
        signal = extract_eeg_channels(df)
        
        # --- PROSES ARTIFACT REMOVAL DILAKUKAN DI SINI ---
        if config.get('remove_eye_artifacts', True) and signal is not None:
            signal = remove_ocular_artifacts_ica(signal, EEG_CHANNELS, config['ica_threshold'])
        return signal
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def compute_hilbert_spectrum(eeg_signal, config):
    """
    Menghasilkan Hilbert Spectrum (Waktu x Frekuensi = Energi)
    Output shape: (n_frames, n_channels * n_freq_bins)
    """
    n_samples, n_channels = eeg_signal.shape
    fs = config['sample_rate']
    f_min = config['f_min']
    f_max = config['f_max']
    n_bins = config['n_freq_bins']
    hop_length = config['hop_length']
    win_length = config['win_length']
    num_imfs = config['num_imfs']
    
    # 1. Buat batas keranjang (bins) frekuensi
    freq_edges = np.linspace(f_min, f_max, n_bins + 1)
    
    ceemdan = CEEMDAN(trials=config['ceemdan_trials'], noise_scale=0.2, parallel=False)
    all_channel_spectra = []
    
    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float64)
        
        # Ekstrak IMFs
        imfs = ceemdan(signal, max_imf=num_imfs)
        actual_imfs = imfs.shape[0]
        if actual_imfs > num_imfs:
            imfs = imfs[:num_imfs, :]
        
        current_n_samples = n_samples
            
        # Matriks kosong untuk Hilbert Spectrum per channel
        hilbert_spec = np.zeros((n_bins, n_samples))
        
        for i in range(imfs.shape[0]):
            analytic_signal = hilbert(imfs[i])
            
            amp = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            freq = (np.diff(phase) / (2.0*np.pi) * fs)
            freq = np.insert(freq, 0, freq[0])
            
            # 2. Binning frekuensi
            bin_indices = np.digitize(freq, freq_edges) - 1
            
            # 3. Akumulasi Energi
            for t in range(n_samples):
                b = bin_indices[t]
                if 0 <= b < n_bins:
                    hilbert_spec[b, t] += (amp[t] ** 2) 
        
        # ---------------------------------------------------------
        # PADDING & WINDOWING DENGAN OVERLAP
        # ---------------------------------------------------------
        if current_n_samples > win_length:
            remainder = (current_n_samples - win_length) % hop_length
            if remainder > 0:
                pad_length = hop_length - remainder
                # Pad dengan angka nol di bagian ekor (axis ke-1/waktu)
                hilbert_spec = np.pad(hilbert_spec, ((0, 0), (0, pad_length)), mode='constant')
                current_n_samples += pad_length

        if current_n_samples < win_length:
            n_frames = 0
            framed_spec = np.zeros((n_bins, 0)) # Mencegah error jika data terlalu pendek
        else:
            n_frames = 1 + (current_n_samples - win_length) // hop_length
            framed_spec = np.zeros((n_bins, n_frames))
            
            for t_idx in range(n_frames):
                start = t_idx * hop_length
                end = start + win_length  # Menggunakan win_length agar terjadi overlap
                framed_spec[:, t_idx] = np.mean(hilbert_spec[:, start:end], axis=1)  
        
        all_channel_spectra.append(framed_spec)
        
    all_channel_spectra = np.array(all_channel_spectra)
    
    # Transpose & Flatten
    features_transposed = all_channel_spectra.transpose(2, 0, 1)
    features_flat = features_transposed.reshape(features_transposed.shape[0], -1)
    
    # Log transform stabilitas
    features_flat = np.log(features_flat + 1e-9)
    
    # Normalisasi z-score per kalimat
    # Menghitung mean dan standar deviasi sepanjang sumbu waktu (axis=0)
    mean_val = np.mean(features_flat, axis=0)
    std_val = np.std(features_flat, axis=0)
    
    # Mencegah error pembagian dengan nol menggunakan 1e-6
    features_flat = (features_flat - mean_val) / (std_val + 1e-6)
    
    return features_flat.astype(np.float32)

# ============================================================================
# DATA SPLIT & PREPROCESSING
# ============================================================================

def split_dataset_by_sentence(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    unique_sentences = df['sentence'].unique()
    n_unique = len(unique_sentences)
    
    train_count = int(n_unique * train_ratio)
    val_count = int(n_unique * val_ratio)
    
    shuffled_sentences = np.random.permutation(unique_sentences)
    train_sentences = set(shuffled_sentences[:train_count])
    val_sentences = set(shuffled_sentences[train_count:train_count+val_count])
    
    df['split'] = df['sentence'].apply(
        lambda x: 'train' if x in train_sentences 
                  else ('val' if x in val_sentences else 'test')
    )
    return df

def load_and_preprocess_dataset(config, target_subject):
    print(f"\n[STEP 1] Load dataset CSV for subject: {target_subject}...")
    df = pd.read_csv(DATASET_CSV)
    
    df = df[df['subject'] == target_subject].copy()
    if len(df) == 0:
        raise ValueError(f"Tidak ada data ditemukan untuk subject: {target_subject}")
    print(f"Total records found for {target_subject}: {len(df)}")
    
    print("[STEP 2] Split dataset (70% train, 10% val, 20% test) by sentence...")
    df = split_dataset_by_sentence(df, config['train_ratio'], config['val_ratio'], config['test_ratio'])
    print(df['split'].value_counts())
    
    print("\n[STEP 3] Load EEG signals and compute Hilbert Spectrum...")
    data = {'train': {'features': [], 'targets': [], 'metadata': []},
            'val': {'features': [], 'targets': [], 'metadata': []},
            'test': {'features': [], 'targets': [], 'metadata': []}}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Hilbert Spectrum"):
        id_val, subject, gender, sentence, split = row['id'], row['subject'], row['gender'], row['sentence'], row['split']
        eeg_signal = load_eeg_signal(id_val, subject, gender, config)
        
        if eeg_signal is None or eeg_signal.shape[0] < config['hop_length']:
            continue
            
        hilbert_features = compute_hilbert_spectrum(eeg_signal, config)
        
        data[split]['features'].append(hilbert_features)
        data[split]['targets'].append(sentence)
        data[split]['metadata'].append({'id': id_val, 'subject': subject, 'gender': gender, 'sentence': sentence})
    
    print(f"\n[SUMMARY] Loaded {len(data['train']['features'])} train, "
          f"{len(data['val']['features'])} val, {len(data['test']['features'])} test")
    return data

# ============================================================================
# DATASET & DATALOADER
# ============================================================================

class EEGDataset(Dataset):
    def __init__(self, features, targets, tokenizer, metadata=None):
        self.features = features
        self.targets = targets
        self.tokenizer = tokenizer
        self.metadata = metadata or [{}] * len(features)
    
    def __len__(self): return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'feature': torch.FloatTensor(self.features[idx]),
            'target': torch.LongTensor(self.tokenizer.text_to_int(self.targets[idx])),
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

# ============================================================================
# EVALUATION METRIC (CER)
# ============================================================================

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

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder=None):
    total_loss, total_cer, num_batches, count = 0, 0, 0, 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 1. Reset model ke mode train
        model.train()
        
        # 2. Paksa eksplisit LSTM masuk ke mode train (Workaround bug inheritance CuDNN)
        if hasattr(model, 'decoder'):
            model.decoder.train()
            if hasattr(model.decoder, 'lstm'):
                model.decoder.lstm.train()
            elif hasattr(model.decoder, 'rnn'):
                model.decoder.rnn.train()
        
        features, feature_length = batch['feature'].to(device), batch['feature_length'].to(device) 
        targets, target_length = batch['target'].to(device), batch['target_length'].to(device) 
        
        optimizer.zero_grad()
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
        logits = model.joiner.output_proj(joint) 
        
        enc_out_lengths = model.get_encoder_out_lengths(feature_length)
        loss = F.rnnt_loss(
            logits=logits, targets=targets.to(torch.int32),
            logit_lengths=enc_out_lengths.to(torch.int32),
            target_lengths=target_length.to(torch.int32), blank=0
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if beam_decoder is not None:
            for i in range(features.shape[0]):
                sample_eeg = features[i:i+1]
                pred_text = beam_decoder.decode(sample_eeg)
                target_text = tokenizer.int_to_text(targets[i].cpu().numpy().tolist())
                total_cer += compute_cer(target_text, pred_text)
        
        total_loss += loss.item()
        num_batches += 1
        count += len(targets)
    
    return (total_loss / num_batches) if num_batches > 0 else 0, (total_cer / count) if count > 0 else 0

def evaluate(model, loader, tokenizer, device, beam_decoder=None, desc="Evaluating"):
    model.eval()
    total_loss, total_cer, count = 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            features, feature_length = batch['feature'].to(device), batch['feature_length'].to(device)
            targets, target_length = batch['target'].to(device), batch['target_length'].to(device)
            
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
            logits = model.joiner.output_proj(joint)
            
            enc_out_lengths = model.get_encoder_out_lengths(feature_length)
            loss = F.rnnt_loss(
                logits=logits, targets=targets.to(torch.int32),
                logit_lengths=enc_out_lengths.to(torch.int32),
                target_lengths=target_length.to(torch.int32), blank=0
            )
            total_loss += loss.item()
            
            if beam_decoder:
                for i in range(features.shape[0]):
                    pred_text = beam_decoder.decode(features[i:i+1])
                    target_text = tokenizer.int_to_text(targets[i].cpu().numpy().tolist())
                    total_cer += compute_cer(target_text, pred_text)
            count += len(targets)
            
    return (total_loss / len(loader)) if len(loader) > 0 else 0, (total_cer / count) if count > 0 else 1.0

def train(model, train_loader, val_loader, tokenizer, config, device, target_subject):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    beam_decoder = beam_decoder_char.BeamDecoderChar(model, tokenizer, beam_size=3, max_sym_per_frame=15)
    
    history = {'train_loss': [], 'train_cer': [], 'val_loss': [], 'val_cer': []}
    
    # Simpan model dengan nama subjek
    best_model_path = os.path.join(OUTPUT_DIR, f'{target_subject}_hilbert_best_model_5_0.pt')
    best_cer = float('inf')
    
    print("\n[STEP 5] Training model...")
    for epoch in range(config['num_epochs']):
        print(f"\n[Epoch {epoch+1}/{config['num_epochs']}]")
        train_loss, train_cer = train_epoch(model, train_loader, optimizer, tokenizer, device, beam_decoder)
        val_loss, val_cer = evaluate(model, val_loader, tokenizer, device, beam_decoder, desc="Validating")
        
        history['train_loss'].append(train_loss)
        history['train_cer'].append(train_cer)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        
        print(f"Train Loss: {train_loss:.4f} | Train CER: {train_cer:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val CER:   {val_cer:.4f}")
        
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'config': config, 'cer': best_cer},
                       best_model_path)
            print(f"  --> [SAVE] New Best Validation CER: {best_cer:.4f}. Model saved.")

    if os.path.exists(best_model_path):
        print("\n[FINAL] Loading best model for testing...")
        model.load_state_dict(torch.load(best_model_path, weights_only=False)['model_state_dict'])
        
    return history, beam_decoder

def predict_and_save_csv(model, test_loader, tokenizer, output_dir, device, beam_decoder, target_subject):
    model.eval()
    predictions_list = []
    
    print("\n[STEP 6] Predicting on Test Set & Generating CSV...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            features = batch['feature'].to(device)
            metadata_batch = batch['metadata']
            
            for i, meta in enumerate(metadata_batch):
                ground_truth = meta['sentence']
                pred_text = beam_decoder.decode(features[i:i+1]) if beam_decoder else ""
                cer = compute_cer(ground_truth, pred_text)
                
                predictions_list.append({
                    'id': meta['id'], 'subject': meta['subject'], 'gender': meta['gender'],
                    'sentence': ground_truth, 'prediction': pred_text, 'cer': cer
                })
                
    predictions_df = pd.DataFrame(predictions_list)
    csv_path = os.path.join(output_dir, f'{target_subject}_hilbert_test_predictions_5_0.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Test predictions saved to {csv_path}")
    print(f"Average Test CER: {predictions_df['cer'].mean():.4f}")
    return predictions_df

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_history(history, output_dir, target_subject):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    axes[0].set_title(f'Loss History ({target_subject} - Hilbert Spectrum)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, history['train_cer'], 'b-o', label='Train CER')
    axes[1].plot(epochs, history['val_cer'], 'r-s', label='Val CER')
    axes[1].set_title(f'CER History ({target_subject} - Hilbert Spectrum)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Character Error Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{target_subject}_hilbert_training_history_5_0.png'), dpi=300)
    plt.close()

# ============================================================================
# MAIN EXECUTOR
# ============================================================================

def main():
    # ---------------------------------------------------------
    # TENTUKAN SUBJEK DI SINI
    # ---------------------------------------------------------
    TARGET_SUBJECT = 'SUB1' 
    
    print("=" * 80)
    print(f"EEG-to-Text Training Pipeline (Subject: {TARGET_SUBJECT} | Hilbert Spectrum Features)")
    print(f"[INFO] Using device: {DEVICE}") # <--- Pindah ke sini agar bebas spam
    print("=" * 80)
    
    data = load_and_preprocess_dataset(CONFIG, TARGET_SUBJECT)
    
    print("\n[STEP 4] Build Character Tokenizer...")
    all_texts = data['train']['targets'] + data['val']['targets'] + data['test']['targets']
    tokenizer = CharTokenizer(transcripts=all_texts)
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    print(f"Vocab size: {CONFIG['vocab_size']}")
    
    train_dataset = EEGDataset(data['train']['features'], data['train']['targets'], tokenizer, data['train']['metadata'])
    val_dataset = EEGDataset(data['val']['features'], data['val']['targets'], tokenizer, data['val']['metadata'])
    test_dataset = EEGDataset(data['test']['features'], data['test']['targets'], tokenizer, data['test']['metadata'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch) 
    
    model = ConformerTransducer(CONFIG).to(DEVICE)
    
    history, beam_decoder = train(model, train_loader, val_loader, tokenizer, CONFIG, DEVICE, TARGET_SUBJECT)
    
    with open(os.path.join(OUTPUT_DIR, f'{TARGET_SUBJECT}_hilbert_training_history_5_0.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    plot_training_history(history, OUTPUT_DIR, TARGET_SUBJECT)
    predict_and_save_csv(model, test_loader, tokenizer, OUTPUT_DIR, DEVICE, beam_decoder, TARGET_SUBJECT)
    
    print("\n" + "=" * 80)
    print(f"✓ FULL TRAINING PIPELINE COMPLETED FOR {TARGET_SUBJECT}")
    print("=" * 80)

if __name__ == '__main__':
    main()