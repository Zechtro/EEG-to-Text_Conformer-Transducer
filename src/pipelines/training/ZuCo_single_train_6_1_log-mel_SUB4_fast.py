import os
import sys
import re
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
import torchaudio.transforms as T
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
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

EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                   'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

EMOTIV_CHANNEL_INDICES = [
    2,   # AF3  (A3)
    6,   # F7   (A7)
    4,   # F3   (A5)
    8,   # FC5  (A9)
    14,  # T7   (A15)
    22,  # P7   (A23)
    26,  # O1   (A27)
    62,  # O2   (B31)
    58,  # P8   (B27)
    50,  # T8   (B19)
    42,  # FC6  (B11)
    38,  # F4   (B7)
    40,  # F8   (B9)
    34,  # AF4  (B3)
]

CONFIG = {
    'input_dim': 14 * 64,
    'encoder_dim': 128,
    'decoder_dim': 128,
    'joint_dim': 128,
    'vocab_size': None,

    'batch_size': 7,
    'num_epochs': 1500,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,

    'encoder_dropout': 0.2,
    'decoder_dropout': 0.2,

    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,

    'sample_rate': 500,
    'n_fft': 256,
    'win_length': 256,
    'hop_length': 32,
    'n_mels': 64,
    'f_min': 0.5,
    'f_max': 45.0,

    'train_ratio': 0.70,
    'val_ratio': 0.10,
    'test_ratio': 0.20,
    'random_seed': 42,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        if raw.shape[0] != 105:
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
            if raw.shape[1] == 105:
                pass
            elif raw.shape[0] == 105:
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

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.6):
    frontal_indices = [i for i, ch in enumerate(ch_names)
                       if 'AF3' in ch.upper() or 'AF4' in ch.upper()]
    if not frontal_indices:
        return eeg_signal
    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try:
        components = ica.fit_transform(eeg_signal)
    except Exception:
        return eeg_signal
    bad = []
    for i in range(components.shape[1]):
        for f_idx in frontal_indices:
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                bad.append(i)
                break
    if bad:
        components[:, bad] = 0.0
    return ica.inverse_transform(components)

def compute_logmel_spectrogram(eeg_signal, config):
    signal_tensor = torch.FloatTensor(eeg_signal.T)
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        f_min=config['f_min'],
        f_max=config['f_max'],
        n_mels=config['n_mels'],
        power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    mel_spec = mel_transform(signal_tensor)
    log_mel = db_transform(mel_spec).numpy()
    log_mel = log_mel.transpose(2, 0, 1)
    n_frames = log_mel.shape[0]
    features = log_mel.reshape(n_frames, -1)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / (std + 1e-6)
    return features.astype(np.float32)

def load_and_preprocess_dataset(config):
    print(f"\n[STEP 1] Loading ZuCo NR data — subject: {SUBJECT_ID} ({SUBJECT_VERSION})")
    sentences, eeg_arrays = load_zuco_subject(SUBJECT_ID, SUBJECT_VERSION)
    print(f"  Loaded {len(sentences)} sentences")

    if not sentences:
        raise RuntimeError("No valid sentences loaded from ZuCo mat file.")

    emotiv_indices = EMOTIV_CHANNEL_INDICES
    print(f"  Selected channels: {EMOTIV_CHANNELS} (indices {emotiv_indices})")

    print(f"\n[STEP 2] Splitting dataset 70/10/20 (seed={config['random_seed']})...")
    idx_all = list(range(len(sentences)))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=config['test_ratio'], random_state=config['random_seed'])
    val_frac = config['val_ratio'] / (config['train_ratio'] + config['val_ratio'])
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=config['random_seed'])

    print(f"  Train: {len(idx_train)} | Val: {len(idx_val)} | Test: {len(idx_test)}")

    print(f"\n[STEP 3] Extracting Log-Mel Spectrogram features...")
    data = {'train': {}, 'val': {}, 'test': {}}
    for split_name, indices in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        feats, tgts, meta = [], [], []
        for i in tqdm(indices, desc=f"  {split_name}"):
            eeg = eeg_arrays[i][:, emotiv_indices]
            sentence = sentences[i]
            if eeg.shape[0] < config['n_fft']:
                continue
            if config.get('remove_eye_artifacts', True):
                eeg = remove_ocular_artifacts_ica(eeg, EMOTIV_CHANNELS, config['ica_threshold'])
            feats.append(compute_logmel_spectrogram(eeg, config))
            tgts.append(sentence)
            meta.append({'sentence_idx': i, 'subject': SUBJECT_ID,
                         'version': SUBJECT_VERSION, 'sentence': sentence})
        data[split_name]['features'] = feats
        data[split_name]['targets'] = tgts
        data[split_name]['metadata'] = meta

    print(f"\n[SUMMARY] {len(data['train']['features'])} train | "
          f"{len(data['val']['features'])} val | "
          f"{len(data['test']['features'])} test")
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
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_log-mel_fast_model_6_1.pt')

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
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_log-mel_fast_test_predictions_6_1.csv')
    df.to_csv(csv_path, index=False)
    print(f'[SAVE] Test predictions → {csv_path}')
    return train_cer, val_cer, test_cer

def plot_training_history(history):
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax.set_title(f'Loss History (ZuCo {SUBJECT_ID} NR — LogMel Fast)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_log-mel_fast_training_history_6_1.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f'[SAVE] Plot → {path}')

def main():
    print('=' * 80)
    print(f'ZuCo EEG-to-Text Training (FAST) — Subject: {SUBJECT_ID} ({SUBJECT_VERSION}) | NR | Log-Mel')
    print(f'[INFO] Device: {DEVICE}')
    print('=' * 80)

    data = load_and_preprocess_dataset(CONFIG)

    print('\n[STEP 4] Build / load character tokenizer...')
    tok_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_log-mel_fast_char_tokenizer_6_1.pkl')
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

    history = train(model, train_loader, val_loader, CONFIG, DEVICE)

    hist_path = os.path.join(
        OUTPUT_DIR, f'ZuCo_{SUBJECT_ID}_NR_log-mel_fast_training_history_6_1.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    plot_training_history(history)
    train_cer, val_cer, test_cer = predict_and_save_csv(
        model, train_loader, val_loader, test_loader, tokenizer, DEVICE)

    print('\n' + '=' * 80)
    print(f'ZuCo TRAINING COMPLETE (FAST) — Subject: {SUBJECT_ID} ({SUBJECT_VERSION})')
    print(f'  Train CER: {train_cer:.4f} | Val CER: {val_cer:.4f} | Test CER: {test_cer:.4f}')
    print(f'  Artifacts in: {OUTPUT_DIR}')
    print('=' * 80)

if __name__ == '__main__':
    main()
