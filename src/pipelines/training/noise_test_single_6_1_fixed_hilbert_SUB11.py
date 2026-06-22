import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
import pickle

from PyEMD import CEEMDAN
from scipy.signal import hilbert
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

SUBJECT = 'SUB11'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
TEST_CSV = os.path.join(PROJECT_ROOT, f'dataset/{SUBJECT}_eq_3_0_test.csv')
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/raw')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'src/pipelines/training')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

EEG_CHANNELS = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 
                'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8', 
                'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']

CONFIG = {
    'input_dim': 14 * 65,
    'encoder_dim': 128,
    'decoder_dim': 128,
    'joint_dim': 128,
    'vocab_size': None,
    
    'batch_size': 7,
    
    'sample_rate': 256,
    'hop_length': 8,      
    'win_length': 16,
    'f_min': 0.2,
    'f_max': 45.0,
    
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,  
    
    'start_imf': 2,
    'ceemdan_trials': 15,  
    'n_freq_bins': 65,     
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_PATH = os.path.join(OUTPUT_DIR, f'{SUBJECT}_eq_3_0_fixed_hilbert_char_tokenizer_6_1.pkl')
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, f'{SUBJECT}_eq_3_0_fixed_hilbert_best_model_6_1.pt')
OUTPUT_CSV_NAME = f'NOISE_BASELINE_{SUBJECT}_eq_3_0_fixed_hilbert_test_predictions_6_1.csv'

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

def compute_hilbert_spectrum(eeg_signal, config):
    n_samples, n_channels = eeg_signal.shape
    fs, f_min, f_max, n_bins = config['sample_rate'], config['f_min'], config['f_max'], config['n_freq_bins']
    hop_length, win_length = config['hop_length'], config['win_length']
    start_imf = config.get('start_imf', 2)
    freq_edges = np.linspace(f_min, f_max, n_bins + 1)
    
    ceemdan = CEEMDAN(trials=config['ceemdan_trials'], noise_scale=0.2, parallel=False)
    all_channel_spectra = []
    
    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float64)
        imfs = ceemdan(signal)
        imfs = imfs[start_imf:] if start_imf < imfs.shape[0] else imfs[-1:]
        
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
    mean_val, std_val = np.mean(features_flat, axis=0), np.std(features_flat, axis=0)
    features_flat = (features_flat - mean_val) / (std_val + 1e-6)
    
    return features_flat.astype(np.float32)

def process_test_df(df, config):
    features, targets, metadata = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing Test Hilbert Spectrum"):
        id_val, subject, gender, sentence = row['id'], row['subject'], row['gender'], row['sentence']
        eeg_signal = load_eeg_signal(id_val, subject, gender, config)
        if eeg_signal is None or eeg_signal.shape[0] < config['hop_length']: continue
        hilbert_features = compute_hilbert_spectrum(eeg_signal, config)
        features.append(hilbert_features)
        targets.append(sentence)
        metadata.append({'id': id_val, 'subject': subject, 'gender': gender, 'sentence': sentence})
    return features, targets, metadata

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

def predict_on_noise_and_save(model, test_loader, tokenizer, output_dir, device, beam_decoder):
    model.eval()
    predictions_list = []
    
    print("\n" + "="*80)
    print("MENGUJI MODEL MENGGUNAKAN RANDOM GAUSSIAN NOISE (JO ET AL. BASELINE)")
    print("="*80)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing with Noise"):

            real_features = batch['feature'].to(device)
            metadata_batch = batch['metadata']
            
            noise_features = torch.randn_like(real_features)
            
            features = noise_features
            
            for i, meta in enumerate(metadata_batch):
                ground_truth = meta['sentence']
                
                pred_text = beam_decoder.decode(features[i:i+1]) if beam_decoder else ""
                cer = compute_cer(ground_truth, pred_text)
                
                predictions_list.append({
                    'id': meta['id'], 'subject': meta['subject'], 'gender': meta['gender'],
                    'sentence': ground_truth, 'prediction': pred_text, 'cer': cer
                })
                
    predictions_df = pd.DataFrame(predictions_list)
    csv_path = os.path.join(output_dir, OUTPUT_CSV_NAME)
    predictions_df.to_csv(csv_path, index=False)
    
    print(f"\n[SAVE] Prediksi Noise-Baseline disimpan ke: {csv_path}")
    print(f"Rata-rata CER pada Noise murni: {predictions_df['cer'].mean():.4f}")
    
    return predictions_df

def main():
    print("=" * 80)
    print(f"NOISE-BASELINE DIAGNOSTIC ({SUBJECT} | Hilbert Spectrum | Char-Transducer)")
    print(f"[INFO] Using device: {DEVICE}") 
    print("=" * 80)
    
    print("\n[STEP 1] Memuat Tokenizer...")
    if not os.path.exists(TOKENIZER_PATH):
        print(f"❌ [ERROR] Tokenizer tidak ditemukan di {TOKENIZER_PATH}")
        return
        
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
        
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    print(f"✓ Tokenizer berhasil dimuat. Vocab size: {CONFIG['vocab_size']}")
    
    print(f"\n[STEP 2] Memuat Test Set (Ekstraksi fitur Hilbert)...")
    df_test = pd.read_csv(TEST_CSV)
    features_test, targets_test, metadata_test = process_test_df(df_test, CONFIG)
    test_dataset = EEGDataset(features_test, targets_test, tokenizer, metadata_test)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)
    
    print("\n[STEP 3] Membangun model dan memuat bobot terbaik (Best Weights)...")
    model = ConformerTransducer(CONFIG).to(DEVICE)
    
    if os.path.exists(BEST_MODEL_PATH):
        saved_data = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved_data['model_state_dict'], strict=False)
        print(f"✓ Bobot model berhasil dimuat dari: {BEST_MODEL_PATH}")
    else:
        print(f"❌ [ERROR] File model {BEST_MODEL_PATH} tidak ditemukan!")
        return
        
    beam_decoder = beam_decoder_char.BeamDecoderChar(model, tokenizer, beam_size=3, max_sym_per_frame=15)
    
    predict_on_noise_and_save(model, test_loader, tokenizer, OUTPUT_DIR, DEVICE, beam_decoder)

if __name__ == '__main__':
    main()