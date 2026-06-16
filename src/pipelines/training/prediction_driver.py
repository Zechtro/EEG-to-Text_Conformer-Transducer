"""
Script Prediksi/Inferensi EEG-to-Text Conformer-Transducer
==========================================================
Lokasi eksekusi: /src/pipelines/training/
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import pickle

# Import library untuk Hilbert Spectrum
from PyEMD import CEEMDAN
from scipy.signal import hilbert
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ============================================================================
# 1. PENGATURAN NAMA FILE INPUT & OUTPUT (UBAH DI SINI)
# ============================================================================
# Asumsi: script ini dijalankan di /src/pipelines/training/
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '../../../'))

# --- SILAKAN UBAH 3 VARIABEL INI SESUAI NAMA FILE ANDA ---
TOKENIZER_FILE = "SUB1_fixed_hilbert_char_tokenizer_6_1.pkl"
MODEL_FILE = "SUB1_fixed_hilbert_best_model_6_1.pt"
TEST_CSV_FILE = "all_eq_3_0_test.csv" # Berada di folder /dataset

# Path Output
OUTPUT_CSV_FILE = f"SUB1_fixed_hilbert_test_predictions_6_1_all_eq_3.csv"

# Absolute Paths
PATH_TOKENIZER = os.path.join(CURRENT_DIR, TOKENIZER_FILE)
PATH_MODEL = os.path.join(CURRENT_DIR, MODEL_FILE)
PATH_TEST_CSV = os.path.join(PROJECT_ROOT, 'dataset', TEST_CSV_FILE)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
PATH_OUTPUT = os.path.join(CURRENT_DIR, OUTPUT_CSV_FILE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import Model
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

# ============================================================================
# 2. UTILITY FUNCTIONS & FEATURE EXTRACTION (HILBERT SPECTRUM)
# ============================================================================

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.6):
    frontal_indices = [i for i, ch in enumerate(ch_names) if 'AF3' in ch or 'AF4' in ch]
    if not frontal_indices: return eeg_signal 

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
                is_artifact = True; break
        if is_artifact: bad_components.append(i)

    if bad_components: components[:, bad_components] = 0.0
    return ica.inverse_transform(components)

def extract_eeg_channels(eeg_df):
    if all(ch in eeg_df.columns for ch in EEG_CHANNELS):
        return eeg_df[EEG_CHANNELS].values
    raise ValueError("Not all channels found in CSV")

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
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def compute_hilbert_spectrum(eeg_signal, config):
    n_samples, n_channels = eeg_signal.shape
    fs = config['sample_rate']
    f_min, f_max, n_bins = config['f_min'], config['f_max'], config['n_freq_bins']
    hop_length, win_length = config['hop_length'], config['win_length']
    
    start_imf = config.get('start_imf', 2)
    freq_edges = np.linspace(f_min, f_max, n_bins + 1)
    ceemdan = CEEMDAN(trials=config['ceemdan_trials'], noise_scale=0.2, parallel=False)
    all_channel_spectra = []
    
    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float64)
        imfs = ceemdan(signal)
        
        if start_imf < imfs.shape[0]: imfs = imfs[start_imf:]
        else: imfs = imfs[-1:]
        
        current_n_samples = n_samples
        hilbert_spec = np.zeros((n_bins, n_samples))
        
        for i in range(imfs.shape[0]):
            analytic_signal = hilbert(imfs[i])
            amp, phase = np.abs(analytic_signal), np.unwrap(np.angle(analytic_signal))
            freq = np.insert((np.diff(phase) / (2.0*np.pi) * fs), 0, 0)
            
            bin_indices = np.digitize(freq, freq_edges) - 1
            for t in range(n_samples):
                b = bin_indices[t]
                if 0 <= b < n_bins: hilbert_spec[b, t] += (amp[t] ** 2) 
        
        if current_n_samples > win_length:
            rem = (current_n_samples - win_length) % hop_length
            if rem > 0:
                pad = hop_length - rem
                hilbert_spec = np.pad(hilbert_spec, ((0, 0), (0, pad)), mode='constant')
                current_n_samples += pad

        if current_n_samples < win_length:
            n_frames, framed_spec = 0, np.zeros((n_bins, 0)) 
        else:
            n_frames = 1 + (current_n_samples - win_length) // hop_length
            framed_spec = np.zeros((n_bins, n_frames))
            for t_idx in range(n_frames):
                start = t_idx * hop_length
                framed_spec[:, t_idx] = np.mean(hilbert_spec[:, start:start+win_length], axis=1)  
        
        all_channel_spectra.append(framed_spec)
        
    all_channel_spectra = np.array(all_channel_spectra).transpose(2, 0, 1)
    features_flat = np.log(all_channel_spectra.reshape(all_channel_spectra.shape[0], -1) + 1e-9)
    return ((features_flat - np.mean(features_flat, axis=0)) / (np.std(features_flat, axis=0) + 1e-6)).astype(np.float32)

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
# 3. PREDICTION PIPELINE
# ============================================================================

def main():
    print("=" * 80)
    print("EEG-to-Text Prediction/Inference Script")
    print(f"[INFO] Using device: {DEVICE}")
    print("=" * 80)
    
    # 1. LOAD TOKENIZER
    if not os.path.exists(PATH_TOKENIZER):
        raise FileNotFoundError(f"Tokenizer tidak ditemukan di: {PATH_TOKENIZER}")
    print(f"Loading Tokenizer dari: {TOKENIZER_FILE}")
    with open(PATH_TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)
        
    # 2. LOAD MODEL & CONFIG
    if not os.path.exists(PATH_MODEL):
        raise FileNotFoundError(f"Model file tidak ditemukan di: {PATH_MODEL}")
    print(f"Loading Model & Config dari: {MODEL_FILE}")
    
    # Memuat file checkpoint (.pt)
    checkpoint = torch.load(PATH_MODEL, map_location=DEVICE, weights_only=False)
    
    # Otomatis mengekstrak CONFIG yang disimpan saat training
    config = checkpoint['config']
    print(f"Config berhasil diekstrak (Vocab size: {config.get('vocab_size')})")
    
    # Inisiasi arsitektur model dan muat bobot (weights)
    model = ConformerTransducer(config).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # PENTING: Set model ke mode evaluasi
    
    # Inisiasi Beam Decoder
    beam_decoder = beam_decoder_char.BeamDecoderChar(model, tokenizer, beam_size=3, max_sym_per_frame=15)
    
    # 3. LOAD TEST DATA
    if not os.path.exists(PATH_TEST_CSV):
        raise FileNotFoundError(f"File Test CSV tidak ditemukan di: {PATH_TEST_CSV}")
    print(f"Loading Test Dataset dari: {TEST_CSV_FILE}")
    df_test = pd.read_csv(PATH_TEST_CSV)
    
    # 4. INFERENCE LOOP
    predictions_list = []
    
    print("\nMengeksekusi Ekstraksi Fitur & Prediksi Model...")
    with torch.no_grad():
        for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Predicting"):
            id_val, subject, gender, ground_truth = row['id'], row['subject'], row['gender'], row['sentence']
            
            # Load & Ekstrak Fitur
            eeg_signal = load_eeg_signal(id_val, subject, gender, config)
            if eeg_signal is None or eeg_signal.shape[0] < config['hop_length']:
                print(f"[WARNING] Skipping ID {id_val}: Sinyal tidak valid/terlalu pendek.")
                continue
                
            features = compute_hilbert_spectrum(eeg_signal, config)
            
            # Ubah ke tensor dan tambahkan dimensi batch (1, T_frames, Features)
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(DEVICE)
            
            # Decode dengan Beam Search
            pred_text = beam_decoder.decode(features_tensor)
            
            # Hitung Character Error Rate (CER)
            cer = compute_cer(ground_truth, pred_text)
            
            predictions_list.append({
                'id': id_val, 
                'subject': subject, 
                'gender': gender,
                'ground_truth': ground_truth, 
                'prediction': pred_text, 
                'cer': cer
            })
            
    # 5. SIMPAN HASIL PREDIKSI
    df_result = pd.DataFrame(predictions_list)
    df_result.to_csv(PATH_OUTPUT, index=False)
    
    print("\n" + "=" * 80)
    print(f"✓ PREDIKSI SELESAI")
    print(f"Total Sampel Diproses : {len(df_result)}")
    print(f"Rata-rata Test CER    : {df_result['cer'].mean():.4f}")
    print(f"File Hasil Disimpan   : {PATH_OUTPUT}")
    print("=" * 80)

if __name__ == '__main__':
    main()