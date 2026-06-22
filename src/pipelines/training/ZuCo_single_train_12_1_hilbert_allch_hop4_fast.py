import os
import sys
import re
import multiprocessing
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchaudio.functional as F
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
from PyEMD import CEEMDAN
from scipy.signal import hilbert
import pickle
import scipy.io as sio
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

SUBJECT_ID = 'ZGW'
SUBJECT_VERSION = 'v1'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
ZUCO_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/zuco')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'experiments/ZuCo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

# All 105 BioSemi channels are used.

# positions in the 105-channel array (identified from the 14-ch EMOTIV mapping:
#   AF3 → index 2, AF4 → index 34 in the full 105-ch rawData array).
N_CHANNELS = 105
FRONTAL_CH_INDICES_IN_105 = [2, 34]   # AF3, AF4

CONFIG = {
    'input_dim': 105 * 65,  # 6825: all 105 channels × 65 freq bins
    'encoder_dim': 128,
    'decoder_dim': 128,
    'joint_dim': 128,
    'vocab_size': None,

    'batch_size': 4,   # hop=4 → 4× longer sequences → larger ConvSubsampling tensors
    'num_epochs': 1500,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,

    'encoder_dropout': 0.3,
    'decoder_dropout': 0.3,

    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,
    'ica_n_components': 30,

    # hop=4 → 125 Hz frame rate → after ConvSubsampling(4×) → ~31 Hz encoder output
    # → T/U ≈ 5:1 for a typical sentence (sweet spot for RNN-T blank alignment)
    'sample_rate': 500,
    'hop_length': 4,
    'win_length': 32,
    'f_min': 0.2,
    'f_max': 45.0,
    'n_freq_bins': 65,
    'start_imf': 2,
    'ceemdan_trials': 15,

    'train_ratio': 0.70,
    'val_ratio': 0.10,
    'test_ratio': 0.20,
    'random_seed': 42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_EXTRACT_JOBS = 32

# Cache file for hop=4 features — separate from v11_x (hop=16) cache.
FEATURE_CACHE_PATH = os.path.join(
    OUTPUT_DIR,
    f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_features_cache.pkl'
)

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[-–—]', ' ', text)
    text = re.sub(r"[^a-z0-9\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_zuco_sentences_scipy(filepath):
    mat = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data = mat.get('sentenceData', mat.get('data', None))
    if data is None:
        raise KeyError(f"No 'sentenceData' key in {filepath}.")

    sentences, eeg_arrays = [], []
    for trial in (data if hasattr(data, '__iter__') else [data]):
        content = getattr(trial, 'content', None)
        if content is None:
            continue
        sentence = normalize_text(str(content))
        if not sentence:
            continue
        raw = getattr(trial, 'rawData', None)
        if raw is None or not isinstance(raw, np.ndarray) or raw.ndim != 2:
            continue
        if raw.shape[0] != N_CHANNELS:
            continue
        eeg_arrays.append(raw.T.astype(np.float32))
        sentences.append(sentence)

    return sentences, eeg_arrays

def load_zuco_sentences_h5py(filepath):
    import h5py
    sentences, eeg_arrays = [], []
    with h5py.File(filepath, 'r') as f:
        sd = f['sentenceData']
        n_sents = sd['content'].shape[0]
        for i in range(n_sents):
            try:
                chars = f[sd['content'][i, 0]][()].flatten()
                sentence = normalize_text(''.join(chr(int(c)) for c in chars))
            except Exception:
                continue
            if not sentence:
                continue
            try:
                raw = f[sd['rawData'][i, 0]][()].astype(np.float32)
            except Exception:
                continue
            if raw.ndim != 2:
                continue
            if raw.shape[1] == N_CHANNELS:
                pass
            elif raw.shape[0] == N_CHANNELS:
                raw = raw.T
            else:
                continue
            eeg_arrays.append(raw)
            sentences.append(sentence)
    return sentences, eeg_arrays

def load_zuco_subject(subject_id, version):
    nr_dir = os.path.join(ZUCO_DATA_PATH, version, 'NR')
    filepath = os.path.join(nr_dir, f'results{subject_id}_NR.mat')
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ZuCo mat file not found: {filepath}")
    print(f"  Loading: {os.path.basename(filepath)}")
    try:
        return load_zuco_sentences_scipy(filepath)
    except NotImplementedError:
        return load_zuco_sentences_h5py(filepath)

def remove_ocular_artifacts_ica(eeg_signal, frontal_indices, threshold=0.6,
                                 n_components=30):
    """ICA-based ocular artifact removal. frontal_indices: channel positions to
    check for EOG correlation (in the passed eeg_signal column space)."""
    n_comp = min(eeg_signal.shape[1], n_components)
    ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000, tol=0.01)
    try:
        components = ica.fit_transform(eeg_signal)
    except Exception:
        return eeg_signal
    bad = []
    for i in range(components.shape[1]):
        for f_idx in frontal_indices:
            if f_idx >= eeg_signal.shape[1]:
                continue
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                bad.append(i)
                break
    if bad:
        components[:, bad] = 0.0
    return ica.inverse_transform(components)

def compute_hilbert_spectrum(eeg_signal, config):
    n_samples, n_channels = eeg_signal.shape
    fs = config['sample_rate']
    f_min = config['f_min']
    f_max = config['f_max']
    n_bins = config['n_freq_bins']
    hop_length = config['hop_length']
    win_length = config['win_length']
    start_imf = config.get('start_imf', 2)
    freq_edges = np.linspace(f_min, f_max, n_bins + 1)

    ceemdan = CEEMDAN(trials=config['ceemdan_trials'], noise_scale=0.2, parallel=False)
    all_channel_spectra = []

    for ch_idx in range(n_channels):
        signal = eeg_signal[:, ch_idx].astype(np.float64)
        imfs = ceemdan(signal)

        if start_imf < imfs.shape[0]:
            imfs = imfs[start_imf:]
        else:
            imfs = imfs[-1:]

        current_n_samples = n_samples
        hilbert_spec = np.zeros((n_bins, n_samples))

        for i in range(imfs.shape[0]):
            analytic_signal = hilbert(imfs[i])
            amp = np.abs(analytic_signal)
            phase = np.unwrap(np.angle(analytic_signal))
            freq = np.diff(phase) / (2.0 * np.pi) * fs
            freq = np.insert(freq, 0, freq[0])

            bin_indices = np.digitize(freq, freq_edges) - 1

            valid = (bin_indices >= 0) & (bin_indices < n_bins)
            t_idx = np.where(valid)[0]
            b_idx = bin_indices[valid]
            hilbert_spec[b_idx, t_idx] += amp[valid] ** 2

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
    mean_val = np.mean(features_flat, axis=0)
    std_val = np.std(features_flat, axis=0)
    features_flat = (features_flat - mean_val) / (std_val + 1e-6)

    return features_flat.astype(np.float32)

def _extract_sentence(args):
    """Top-level worker for sentence-level parallel feature extraction."""
    eeg, sentence, meta, config = args
    if eeg.shape[0] < config['win_length']:
        return None
    if config.get('remove_eye_artifacts', True):
        eeg = remove_ocular_artifacts_ica(
            eeg,
            frontal_indices=FRONTAL_CH_INDICES_IN_105,
            threshold=config['ica_threshold'],
            n_components=config.get('ica_n_components', 30),
        )
    feat = compute_hilbert_spectrum(eeg, config)
    return feat, sentence, meta

def load_and_preprocess_dataset(config):

    if os.path.exists(FEATURE_CACHE_PATH):
        print(f"\n[CACHE] Loading pre-extracted features from:\n  {FEATURE_CACHE_PATH}")
        with open(FEATURE_CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    print(f"\n[STEP 1] Loading ZuCo NR data — subject: {SUBJECT_ID} ({SUBJECT_VERSION})")
    sentences, eeg_arrays = load_zuco_subject(SUBJECT_ID, SUBJECT_VERSION)
    print(f"  Loaded {len(sentences)} sentences")
    print(f"  Using ALL {N_CHANNELS} EEG channels")

    if not sentences:
        raise RuntimeError("No valid sentences loaded from ZuCo mat file.")

    print(f"\n[STEP 2] Splitting dataset 70/10/20 (seed={config['random_seed']})...")
    idx_all = list(range(len(sentences)))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=config['test_ratio'], random_state=config['random_seed'])
    val_frac = config['val_ratio'] / (config['train_ratio'] + config['val_ratio'])
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=config['random_seed'])

    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

    print(f"\n[STEP 3] Extracting Hilbert Spectrum features "
          f"({N_EXTRACT_JOBS} parallel workers, serial CEEMDAN per sentence)...")
    print(f"  NOTE: 105-channel CEEMDAN is slow. First run may take several hours.")
    data = {'train': {}, 'val': {}, 'test': {}}

    for split_name, indices in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        feats, tgts, meta = [], [], []
        args_list = [
            (eeg_arrays[i],          # full 105-channel array, no slicing
             sentences[i],
             {'sentence_idx': i, 'subject': SUBJECT_ID,
              'version': SUBJECT_VERSION, 'sentence': sentences[i]},
             config)
            for i in indices
        ]
        with multiprocessing.Pool(processes=N_EXTRACT_JOBS) as pool:
            for result in tqdm(pool.imap(_extract_sentence, args_list),
                               total=len(args_list), desc=f'  {split_name}'):
                if result is not None:
                    feats.append(result[0])
                    tgts.append(result[1])
                    meta.append(result[2])
        data[split_name]['features'] = feats
        data[split_name]['targets'] = tgts
        data[split_name]['metadata'] = meta

    print(f"\n[SUMMARY] {len(data['train']['features'])} train | "
          f"{len(data['val']['features'])} val | "
          f"{len(data['test']['features'])} test")

    print(f"\n[CACHE] Saving features to:\n  {FEATURE_CACHE_PATH}")
    with open(FEATURE_CACHE_PATH, 'wb') as f:
        pickle.dump(data, f)

    return data

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
            'metadata': self.metadata[idx],
        }

def collate_batch(batch):
    features = [item['feature'] for item in batch]
    targets = [item['target'] for item in batch]
    max_f = max(f.shape[0] for f in features)
    padded_f = [torch.nn.functional.pad(f, (0, 0, 0, max_f - f.shape[0])) for f in features]
    max_t = max(len(t) for t in targets)
    padded_t = [torch.nn.functional.pad(t, (0, max_t - len(t))) for t in targets]
    return {
        'feature': torch.stack(padded_f),
        'feature_length': torch.LongTensor([f.shape[0] for f in features]),
        'target': torch.stack(padded_t),
        'target_length': torch.LongTensor([len(t) for t in targets]),
        'metadata': [item['metadata'] for item in batch],
    }

def compute_cer(reference, hypothesis):
    if len(reference) == 0:
        return 1.0 if len(hypothesis) > 0 else 0.0
    d = np.zeros((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1): d[i][0] = i
    for j in range(len(hypothesis) + 1): d[0][j] = j
    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            cost = 0 if reference[i-1] == hypothesis[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(reference)][len(hypothesis)] / len(reference)

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss, num_batches = 0, 0
    for batch in tqdm(train_loader, desc='Training'):
        features = batch['feature'].to(device)
        feature_length = batch['feature_length'].to(device)
        targets = batch['target'].to(device)
        target_length = batch['target_length'].to(device)

        optimizer.zero_grad()
        encoder_out = model.encoder(features)

        batch_size = targets.shape[0]
        blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        decoder_input = torch.cat([blank_col, targets], dim=1)
        hidden_state = model.decoder.init_hidden(batch_size, device)
        decoder_out, _ = model.decoder(decoder_input, hidden_state)

        enc_proj = model.joiner.encoder_proj(encoder_out)
        dec_proj = model.joiner.decoder_proj(decoder_out)
        joint = model.joiner.activation(enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1))
        logits = model.joiner.output_proj(joint)

        enc_out_lengths = model.get_encoder_out_lengths(feature_length)
        loss = F.rnnt_loss(
            logits=logits, targets=targets.to(torch.int32),
            logit_lengths=enc_out_lengths.to(torch.int32),
            target_lengths=target_length.to(torch.int32), blank=0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches else 0

def evaluate_loss(model, loader, device, desc='Validating'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            features = batch['feature'].to(device)
            feature_length = batch['feature_length'].to(device)
            targets = batch['target'].to(device)
            target_length = batch['target_length'].to(device)

            encoder_out = model.encoder(features)
            batch_size = targets.shape[0]
            blank_col = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            decoder_input = torch.cat([blank_col, targets], dim=1)
            hidden_state = model.decoder.init_hidden(batch_size, device)
            decoder_out, _ = model.decoder(decoder_input, hidden_state)

            enc_proj = model.joiner.encoder_proj(encoder_out)
            dec_proj = model.joiner.decoder_proj(decoder_out)
            joint = model.joiner.activation(enc_proj.unsqueeze(2) + dec_proj.unsqueeze(1))
            logits = model.joiner.output_proj(joint)

            enc_out_lengths = model.get_encoder_out_lengths(feature_length)
            loss = F.rnnt_loss(
                logits=logits, targets=targets.to(torch.int32),
                logit_lengths=enc_out_lengths.to(torch.int32),
                target_lengths=target_length.to(torch.int32), blank=0,
            )
            total_loss += loss.item()

    return total_loss / len(loader) if len(loader) else 0

def train(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(),
                           lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history = {'train_loss': [], 'val_loss': []}

    model_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_model.pt')

    print('\n[STEP 5] Training model (fast — loss-only validation, saves latest each epoch)...')
    for epoch in range(config['num_epochs']):
        print(f'\n[Epoch {epoch+1}/{config["num_epochs"]}]')
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
        }, model_path)

    print(f'  --> [SAVE] Final model (epoch {config["num_epochs"]}) → {model_path}')
    return history

def predict_split(model, loader, tokenizer, device, decoder, desc):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            features = batch['feature'].to(device)
            for i, meta in enumerate(batch['metadata']):
                gt = meta['sentence']
                pred = decoder.decode(features[i:i+1])
                rows.append({
                    'sentence_idx': meta['sentence_idx'],
                    'subject': meta['subject'],
                    'version': meta['version'],
                    'sentence': gt,
                    'prediction': pred,
                    'cer': compute_cer(gt, pred),
                })
    return rows

def predict_and_save_csv(model, train_loader, val_loader, test_loader, tokenizer, device):
    import pandas as pd
    decoder = beam_decoder_char.BeamDecoderChar(model, tokenizer,
                                                beam_size=3, max_sym_per_frame=15)
    print('\n[STEP 6] Computing CER on all splits...')
    train_rows = predict_split(model, train_loader, tokenizer, device, decoder, 'Train')
    val_rows   = predict_split(model, val_loader,   tokenizer, device, decoder, 'Val')
    test_rows  = predict_split(model, test_loader,  tokenizer, device, decoder, 'Test')

    train_cer = np.mean([r['cer'] for r in train_rows])
    val_cer   = np.mean([r['cer'] for r in val_rows])
    test_cer  = np.mean([r['cer'] for r in test_rows])

    print(f'\n  Average Train CER: {train_cer:.4f}')
    print(f'  Average Val   CER: {val_cer:.4f}')
    print(f'  Average Test  CER: {test_cer:.4f}')

    df = pd.DataFrame(test_rows)
    csv_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_test_predictions.csv')
    df.to_csv(csv_path, index=False)
    print(f'[SAVE] Test predictions → {csv_path}')
    return train_cer, val_cer, test_cer

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_title(f'Loss History (ZuCo {SUBJECT_ID} NR — Hilbert All Channels Fast)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_training_history.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'[SAVE] Plot → {path}')

def main():
    print('=' * 80)
    print(f'ZuCo EEG-to-Text Training (FAST, ALL CHANNELS) — '
          f'Subject: {SUBJECT_ID} ({SUBJECT_VERSION}) | NR | Hilbert')
    print(f'[INFO] Device: {DEVICE}')
    print(f'[INFO] Channels: {N_CHANNELS}  |  input_dim: {CONFIG["input_dim"]}  |  hop_length: {CONFIG["hop_length"]}')
    print('=' * 80)

    data = load_and_preprocess_dataset(CONFIG)

    print('\n[STEP 4] Build / load character tokenizer...')
    tok_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_char_tokenizer.pkl')
    if os.path.exists(tok_path):
        print(f'  Loading existing tokenizer from {tok_path}')
        with open(tok_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        all_texts = (data['train']['targets'] + data['val']['targets']
                     + data['test']['targets'])
        tokenizer = CharTokenizer(transcripts=all_texts)
        with open(tok_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f'  [SAVE] Tokenizer → {tok_path}')

    CONFIG['vocab_size'] = tokenizer.vocab_size()
    print(f'  Vocab size: {CONFIG["vocab_size"]}')

    n_train = len(data['train']['features'])
    n_val   = len(data['val']['features'])
    n_test  = len(data['test']['features'])
    print(f'  Trainable samples: {n_train} train | {n_val} val | {n_test} test')

    train_ds = EEGDataset(data['train']['features'], data['train']['targets'],
                          tokenizer, data['train']['metadata'])
    val_ds = EEGDataset(data['val']['features'], data['val']['targets'],
                        tokenizer, data['val']['metadata'])
    test_ds = EEGDataset(data['test']['features'], data['test']['targets'],
                         tokenizer, data['test']['metadata'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                              shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'],
                            shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, collate_fn=collate_batch)

    model = ConformerTransducer(CONFIG).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n[INFO] Model params: {trainable_params:,} trainable / {total_params:,} total')

    history = train(model, train_loader, val_loader, CONFIG, DEVICE)

    hist_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_hilbert_allch_hop4_training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_history(history)
    train_cer, val_cer, test_cer = predict_and_save_csv(
        model, train_loader, val_loader, test_loader, tokenizer, DEVICE)

    print('\n' + '=' * 80)
    print(f'ZuCo TRAINING COMPLETE — hilbert_allch_hop4 — Subject: {SUBJECT_ID} ({SUBJECT_VERSION})')
    print(f'  Train CER: {train_cer:.4f} | Val CER: {val_cer:.4f} | Test CER: {test_cer:.4f}')
    print(f'  Artifacts in: {OUTPUT_DIR}')
    print('=' * 80)

if __name__ == '__main__':
    main()
