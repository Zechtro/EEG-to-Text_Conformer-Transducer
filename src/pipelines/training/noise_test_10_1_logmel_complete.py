"""
Noise-Baseline Evaluation Script (Log-Mel Spectrogram + IndoGPT Decoder)
=====================================================================
Metodologi: "Are EEG-to-Text Models Working?" (Jo et al., 2024)
Diperbarui dengan Pre-Flight Check dan Looping (SUB1-SUB12, all).
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

# Import library untuk Fitur Log-Mel Spectrogram & Artefak
import torchaudio.transforms as T
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

# Import Tokenizer Bypass
import transformers.utils
import transformers.utils.generic

# Bypass HuggingFace internal checks
if not hasattr(transformers.utils, 'is_tf_available'): transformers.utils.is_tf_available = lambda: False
if not hasattr(transformers.utils.generic, '_is_jax'): transformers.utils.generic._is_jax = lambda x: False
if not hasattr(transformers.utils.generic, '_is_tensorflow'): transformers.utils.generic._is_tensorflow = lambda x: False
if not hasattr(transformers.utils.generic, '_is_numpy'): transformers.utils.generic._is_numpy = lambda x: isinstance(x, np.ndarray)
if not hasattr(transformers.utils.generic, '_is_torch'): transformers.utils.generic._is_torch = lambda x: torch.is_tensor(x)
if not hasattr(transformers.utils.generic, '_is_torch_device'): transformers.utils.generic._is_torch_device = lambda x: isinstance(x, torch.device)

from indobenchmark import IndoNLGTokenizer

warnings.filterwarnings('ignore')

# ============================================================================
# KONFIGURASI PATH GLOBAL & PARAMETER
# ============================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

# Import arsitektur model dan decoder IndoGPT
from model.model import ConformerIndoGPTTransducer
from model.misc.beam_decoder import BeamDecoder

EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

CONFIG = {
    'input_dim': 14 * 64,  
    'encoder_dim': 356,
    'decoder_dim': 768,
    'joint_dim': 768,
    'num_layers': 4,
    'vocab_size': None, # Akan diisi dinamis setelah memuat IndoNLGTokenizer
    
    'batch_size': 7,
    
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,  
    
    'sample_rate': 256,
    'n_fft': 128,          
    'win_length': 128,
    'hop_length': 16,      
    'n_mels': 64,          
    'f_min': 0.5,          
    'f_max': 45.0,         
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# FUNGSI EKSTRAKSI FITUR LOG-MEL SPECTROGRAM
# ============================================================================

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
                is_artifact = True; break
        if is_artifact: bad_components.append(i)
    if bad_components: components[:, bad_components] = 0.0
    return ica.inverse_transform(components)

def extract_eeg_channels(eeg_df):
    if all(ch in eeg_df.columns for ch in EEG_CHANNELS): return eeg_df[EEG_CHANNELS].values
    else: raise ValueError("Not all channels found in CSV")

def load_eeg_signal(id_val, subject, gender, config):
    csv_folder = os.path.join(RAW_DATA_PATH, gender, subject, 'csv')
    if not os.path.isdir(csv_folder): return None
    matching_files = [f for f in os.listdir(csv_folder) if f.startswith(str(id_val) + '_') and f.endswith('.bp.csv')]
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

def compute_logmel_spectrogram(eeg_signal, config):
    signal_tensor = torch.FloatTensor(eeg_signal.T) 
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'], n_fft=config['n_fft'],
        win_length=config['win_length'], hop_length=config['hop_length'],
        f_min=config['f_min'], f_max=config['f_max'], n_mels=config['n_mels'], power=2.0
    )
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    mel_spec = mel_transform(signal_tensor) 
    log_mel = db_transform(mel_spec)
    log_mel_np = log_mel.numpy().transpose(2, 0, 1) 
    n_frames = log_mel_np.shape[0]
    features_flat = log_mel_np.reshape(n_frames, -1)
    
    mean_val = np.mean(features_flat, axis=0)
    std_val = np.std(features_flat, axis=0)
    features_flat = (features_flat - mean_val) / (std_val + 1e-6)
    
    return features_flat.astype(np.float32)

def process_test_df(df, config):
    features, targets, metadata = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  -> Ekstraksi Log-Mel"):
        id_val, subject, gender, sentence = row['id'], row['subject'], row['gender'], row['sentence']
        eeg_signal = load_eeg_signal(id_val, subject, gender, config)
        if eeg_signal is None or eeg_signal.shape[0] < config['n_fft']: continue
        
        logmel_features = compute_logmel_spectrogram(eeg_signal, config)
        
        features.append(logmel_features)
        targets.append(sentence)
        metadata.append({'id': id_val, 'subject': subject, 'gender': gender, 'sentence': sentence})
    return features, targets, metadata

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
        # Gunakan fungsi encode milik HuggingFace
        encoded_tokens = self.tokenizer.encode(self.targets[idx])
        
        # SHIFTING: Geser semua token +1 untuk ruang <blank>
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

# ============================================================================
# FUNGSI INFERENSI PADA NOISE
# ============================================================================

def predict_on_noise_and_save(model, test_loader, tokenizer, output_dir, device, beam_decoder, output_csv_name):
    model.eval()
    predictions_list = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  -> Prediksi Noise"):
            # 1. Ambil matriks fitur Log-Mel Spectrogram asli
            real_features = batch['feature'].to(device)
            metadata_batch = batch['metadata']
            
            # 2. GANTI DENGAN GAUSSIAN NOISE MURNI
            noise_features = torch.randn_like(real_features)
            
            # Timpa fitur yang masuk ke model menjadi acak
            features = noise_features
            
            for i, meta in enumerate(metadata_batch):
                ground_truth = meta['sentence']
                
                # Decode noise murni menjadi teks
                pred_text = beam_decoder.decode(features[i:i+1]) if beam_decoder else ""
                cer = compute_cer(ground_truth, pred_text)
                
                predictions_list.append({
                    'id': meta['id'], 'subject': meta['subject'], 'gender': meta['gender'],
                    'sentence': ground_truth, 'prediction': pred_text, 'cer': cer
                })
                
    predictions_df = pd.DataFrame(predictions_list)
    csv_path = os.path.join(output_dir, output_csv_name)
    predictions_df.to_csv(csv_path, index=False)
    
    print(f"  [SAVE] Disimpan ke: {output_csv_name}")
    print(f"  Rata-rata CER pada Noise murni: {predictions_df['cer'].mean():.4f}")
    
    return predictions_df

# ============================================================================
# MAIN EXECUTOR (PRE-FLIGHT CHECK & LOOP)
# ============================================================================

def main():
    SUBJECTS = [f"SUB{i}" for i in range(1, 13)] + ['all']
    
    print("=" * 80)
    print("PRE-FLIGHT CHECK: MEMERIKSA KELENGKAPAN FILE MODEL & DATA (IndoGPT)")
    print("=" * 80)
    
    # Init Tokenizer sekali saja di luar loop (Karena IndoGPT universal)
    print("\n[STEP 0] Loading Universal IndoNLGTokenizer...")
    tokenizer = IndoNLGTokenizer.from_pretrained("indobenchmark/indogpt")
    
    def dummy_pad(encoded_inputs, **kwargs): return encoded_inputs
    tokenizer.pad = dummy_pad
    
    if not hasattr(tokenizer, 'int_to_text'): tokenizer.int_to_text = tokenizer.decode
        
    CONFIG['vocab_size'] = tokenizer.vocab_size + 1
    
    valid_subjects = []
    missing_reports = []
    
    # 1. Pengecekan Cepat (Dry-Run)
    for subject in SUBJECTS:
        test_csv_path = os.path.join(PROJECT_ROOT, f'dataset/{subject}_eq_3_0_test.csv')
        # Model khusus LogMel + IndoGPT
        best_model_path = os.path.join(OUTPUT_DIR, f'{subject}_eq_3_0_logmel_best_model_10_1_IndoGPT.pt')
        
        missing = []
        if not os.path.exists(test_csv_path): missing.append("Test CSV")
        if not os.path.exists(best_model_path): missing.append("Bobot Model (.pt)")
        
        if missing:
            missing_reports.append(f"  ❌ {subject.upper()}: Kurang file ({', '.join(missing)})")
        else:
            valid_subjects.append(subject)
            print(f"  ✓ {subject.upper()}: Siap dievaluasi")
            
    # Tampilkan Peringatan Jika Ada yang Kurang
    if missing_reports:
        print("\n[WARNING] Beberapa subjek akan DILOMPATI karena file belum lengkap:")
        for report in missing_reports:
            print(report)
            
    # Jika tidak ada satupun yang siap, hentikan program
    if not valid_subjects:
        print("\n❌ TIDAK ADA SUBJEK YANG BISA DIEKSEKUSI. Program dihentikan.")
        return
        
    print(f"\n[INFO] Melanjutkan proses evaluasi untuk {len(valid_subjects)} subjek valid menggunakan {DEVICE}...")
    
    # 2. Eksekusi Utama hanya untuk subjek yang valid
    for subject in valid_subjects:
        print(f"\n" + "=" * 60)
        print(f"Memproses Noise-Baseline untuk Subjek: {subject.upper()} (IndoGPT)")
        print("=" * 60)
        
        test_csv_path = os.path.join(PROJECT_ROOT, f'dataset/{subject}_eq_3_0_test.csv')
        best_model_path = os.path.join(OUTPUT_DIR, f'{subject}_eq_3_0_logmel_best_model_10_1_IndoGPT.pt')
        output_csv_name = f'NOISE_BASELINE_{subject}_eq_3_0_logmel_test_predictions_10_1_IndoGPT.csv'
        
        # Load Test Set & Ekstraksi
        df_test = pd.read_csv(test_csv_path)
        features_test, targets_test, metadata_test = process_test_df(df_test, CONFIG)
        test_dataset = EEGDataset(features_test, targets_test, tokenizer, metadata_test)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
        
        # Bangun Model & Load Weights
        model = ConformerIndoGPTTransducer(CONFIG).to(DEVICE)
        saved_data = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved_data['model_state_dict'], strict=False)
        
        # Evaluasi Noise menggunakan BeamDecoder standar (Bukan Char-Decoder)
        beam_decoder = BeamDecoder(model, tokenizer, beam_size=3)
        predict_on_noise_and_save(model, test_loader, tokenizer, OUTPUT_DIR, DEVICE, beam_decoder, output_csv_name)
        
    print("\n" + "=" * 80)
    print("SELESAI! Seluruh file prediksi Noise-Baseline telah berhasil di-*generate*.")
    print("=" * 80)

if __name__ == '__main__':
    main()