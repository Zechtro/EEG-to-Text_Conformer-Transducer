"""
Full Training Pipeline for EEG-to-Text Conformer-Transducer (Cross-Subject)
============================================================================

Fitur:
1. Membaca dataset Train, Val, dan Test yang sudah displit dari file CSV masing-masing
2. Mengekstraksi seluruh subjek (All Subjects) atau spesifik
3. Ekstraksi fitur menggunakan LOG-MEL SPECTROGRAM (SOTA ASR adaptation)
4. Training menggunakan RNN-T loss dengan evaluasi CER (BeamDecoder)
5. Bebas spam log multiprocessing
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

# Import library untuk Fitur Log-Mel Spectrogram
import torchaudio.transforms as T
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import pickle

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI PATH & PARAMETER
# ============================================================================

SUBJECT = 'SUB5'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
TRAIN_CSV = os.path.join(PROJECT_ROOT, 'dataset/SUB5_eq_3_0_train.csv')
VAL_CSV = os.path.join(PROJECT_ROOT, 'dataset/SUB5_eq_3_0_val.csv')
TEST_CSV = os.path.join(PROJECT_ROOT, 'dataset/SUB5_eq_3_0_test.csv')

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

CONFIG = {
    # input_dim sekarang = 14 channels * 64 n_mels
    'input_dim': 14 * 64,
    'encoder_dim': 128,
    'decoder_dim': 128,
    'joint_dim': 128,
    'vocab_size': None,
    
    'batch_size': 7,
    'num_epochs': 200, 
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,  
    
    'encoder_dropout': 0.2, 
    'decoder_dropout': 0.2, 
    
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,  
    
    # Parameter SOTA Log-Mel Spectrogram untuk EEG
    'sample_rate': 256,
    'n_fft': 128,          # Ukuran window 0.5 detik (menangkap low freq)
    'win_length': 128,
    'hop_length': 16,      # Overlap pergeseran 62.5 ms
    'n_mels': 64,          # Resolusi filterbank frekuensi
    'f_min': 0.5,          # Membuang offset DC (0 Hz)
    'f_max': 45.0,         # Memotong frekuensi di bawah 50Hz (Listrik)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# UTILITY FUNCTIONS & ARTIFACT REMOVAL
# ============================================================================

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.6):
    frontal_indices = [i for i, ch in enumerate(ch_names) if 'AF3' in ch or 'AF4' in ch]
    if not frontal_indices:
        return eeg_signal 

    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try:
        components = ica.fit_transform(eeg_signal) 
    except:
        return eeg_signal 

    bad_components = []
    for i in range(components.shape[1]):
        is_artifact = False
        for f_idx in frontal_indices:
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                is_artifact = True
                break
        if is_artifact:
            bad_components.append(i)

    if bad_components:
        components[:, bad_components] = 0.0

    cleaned_signal = ica.inverse_transform(components)
    return cleaned_signal

def extract_eeg_channels(eeg_df):
    if all(ch in eeg_df.columns for ch in EEG_CHANNELS):
        return eeg_df[EEG_CHANNELS].values
    else:
        raise ValueError("Not all channels found in CSV")

def load_eeg_signal(id_val, subject, gender, config):
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    if not os.path.isdir(csv_folder): return None
    
    matching_files = [f for f in os.listdir(csv_folder) 
                      if f.startswith(str(id_val) + '_') and f.endswith('.bp.csv')]
    if not matching_files: return None
    
    file_path = os.path.join(csv_folder, matching_files[0])
    try:
        df = pd.read_csv(file_path, skiprows=1)
        signal = extract_eeg_channels(df)
        
        if config.get('remove_eye_artifacts', True) and signal is not None:
            signal = remove_ocular_artifacts_ica(signal, EEG_CHANNELS, config['ica_threshold'])
        return signal
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

# ============================================================================
# FEATURE EXTRACTION (LOG-MEL SPECTROGRAM)
# ============================================================================

def compute_logmel_spectrogram(eeg_signal, config):
    """
    Ekstraksi fitur Log-Mel Spectrogram yang menggantikan Hilbert Spectrum.
    Menggunakan torchaudio agar komputasi di CPU/GPU jauh lebih cepat.
    """
    # Torchaudio mengekspektasikan format (Time) atau (Channel, Time)
    # Shape awal eeg_signal: (n_samples, n_channels)
    signal_tensor = torch.FloatTensor(eeg_signal.T) # Shape: (n_channels, n_samples)
    
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        f_min=config['f_min'],
        f_max=config['f_max'],
        n_mels=config['n_mels'],
        power=2.0 # Power Spectrogram
    )
    
    # Konversi amplitudo ke skala Decibel (Log)
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    
    # 1. Hitung Mel Spectrogram
    mel_spec = mel_transform(signal_tensor) # Shape: (n_channels, n_mels, n_frames)
    
    # 2. Konversi ke Log-Mel (Decibel)
    log_mel = db_transform(mel_spec)
    
    # 3. Reformat matriks agar cocok dengan ekspektasi model (n_frames, n_channels * n_mels)
    log_mel_np = log_mel.numpy()
    # Ubah menjadi (n_frames, n_channels, n_mels)
    log_mel_np = log_mel_np.transpose(2, 0, 1) 
    
    n_frames = log_mel_np.shape[0]
    features_flat = log_mel_np.reshape(n_frames, -1)
    
    # 4. CMVN (Cepstral Mean and Variance Normalization) per kalimat
    mean_val = np.mean(features_flat, axis=0)
    std_val = np.std(features_flat, axis=0)
    features_flat = (features_flat - mean_val) / (std_val + 1e-6)
    
    return features_flat.astype(np.float32)

# ============================================================================
# DATA LOADING (MULTI-CSV)
# ============================================================================

def process_split_df(df, split_name, config):
    """Fungsi pembantu untuk memproses setiap dataframe split"""
    features = []
    targets = []
    metadata = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name} Log-Mel"):
        id_val, subject, gender, sentence = row['id'], row['subject'], row['gender'], row['sentence']
        eeg_signal = load_eeg_signal(id_val, subject, gender, config)
        
        if eeg_signal is None or eeg_signal.shape[0] < config['n_fft']:
            continue
            
        # Panggil fungsi ekstraksi Log-Mel
        logmel_features = compute_logmel_spectrogram(eeg_signal, config)
        
        features.append(logmel_features)
        targets.append(sentence)
        metadata.append({'id': id_val, 'subject': subject, 'gender': gender, 'sentence': sentence})
        
    return features, targets, metadata

def load_and_preprocess_dataset(config):
    print(f"\n[STEP 1 & 2] Load pre-split datasets (Train, Val, Test)...")
    df_train = pd.read_csv(TRAIN_CSV)
    df_val = pd.read_csv(VAL_CSV)
    df_test = pd.read_csv(TEST_CSV)
    
    print(f"Total baris - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    print("\n[STEP 3] Load EEG signals and compute Log-Mel Spectrogram...")
    data = {'train': {}, 'val': {}, 'test': {}}
    
    data['train']['features'], data['train']['targets'], data['train']['metadata'] = process_split_df(df_train, 'Train', config)
    data['val']['features'], data['val']['targets'], data['val']['metadata'] = process_split_df(df_val, 'Val', config)
    data['test']['features'], data['test']['targets'], data['test']['metadata'] = process_split_df(df_test, 'Test', config)
    
    print(f"\n[SUMMARY] Berhasil load {len(data['train']['features'])} train, "
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
        model.train()
        
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

def train(model, train_loader, val_loader, tokenizer, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    beam_decoder = beam_decoder_char.BeamDecoderChar(model, tokenizer, beam_size=3, max_sym_per_frame=15)
    
    history = {'train_loss': [], 'train_cer': [], 'val_loss': [], 'val_cer': []}
    
    best_model_path = os.path.join(OUTPUT_DIR, f'{SUBJECT}_eq_3_0_log-mel_best_model_6_1.pt')
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
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=False)['model_state_dict'])
        
    return history, beam_decoder

def predict_and_save_csv(model, test_loader, tokenizer, output_dir, device, beam_decoder):
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
    csv_path = os.path.join(output_dir, f'{SUBJECT}_eq_3_0_log-mel_test_predictions_6_1.csv')
    predictions_df.to_csv(csv_path, index=False)
    print(f"[SAVE] Test predictions saved to {csv_path}")
    print(f"Average Test CER: {predictions_df['cer'].mean():.4f}")
    return predictions_df

# ============================================================================
# PLOTTING
# ============================================================================

def plot_training_history(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    axes[0].set_title(f'Loss History ({SUBJECT} - LogMel Spectrogram)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(epochs, history['train_cer'], 'b-o', label='Train CER')
    axes[1].plot(epochs, history['val_cer'], 'r-s', label='Val CER')
    axes[1].set_title(f'CER History ({SUBJECT} - LogMel Spectrogram)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Character Error Rate')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{SUBJECT}_eq_3_0_log-mel_training_history_6_1.png'), dpi=300)
    plt.close()

# ============================================================================
# MAIN EXECUTOR
# ============================================================================

def main():
    print("=" * 80)
    print(f"EEG-to-Text Training Pipeline ({SUBJECT} | Log-Mel Spectrogram Features)")
    print(f"[INFO] Using device: {DEVICE}") 
    print("=" * 80)
    
    data = load_and_preprocess_dataset(CONFIG)
    
    print("\n[STEP 4] Build or Load Character Tokenizer...")
    tokenizer_path = os.path.join(OUTPUT_DIR, f'{SUBJECT}_eq_3_0_log-mel_char_tokenizer_6_1.pkl')
    
    if os.path.exists(tokenizer_path):
        print(f"Memuat tokenizer yang sudah ada dari: {tokenizer_path}")
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        print("Membangun tokenizer baru dari data...")
        all_texts = data['train']['targets'] + data['val']['targets'] + data['test']['targets']
        tokenizer = CharTokenizer(transcripts=all_texts)
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"[SAVE] Tokenizer berhasil disimpan ke: {tokenizer_path}")
        
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    print(f"Vocab size: {CONFIG['vocab_size']}")
    
    train_dataset = EEGDataset(data['train']['features'], data['train']['targets'], tokenizer, data['train']['metadata'])
    val_dataset = EEGDataset(data['val']['features'], data['val']['targets'], tokenizer, data['val']['metadata'])
    test_dataset = EEGDataset(data['test']['features'], data['test']['targets'], tokenizer, data['test']['metadata'])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch) 
    
    model = ConformerTransducer(CONFIG).to(DEVICE)
    
    history, beam_decoder = train(model, train_loader, val_loader, tokenizer, CONFIG, DEVICE)
    
    with open(os.path.join(OUTPUT_DIR, f'{SUBJECT}_eq_3_0_log-mel_training_history_6_1.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    plot_training_history(history, OUTPUT_DIR)
    predict_and_save_csv(model, test_loader, tokenizer, OUTPUT_DIR, DEVICE, beam_decoder)
    
    print("\n" + "=" * 80)
    print(f"✓ FULL TRAINING PIPELINE COMPLETED FOR {SUBJECT}")
    print("=" * 80)

if __name__ == '__main__':
    main()