import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import scipy.io as sio
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import warnings

warnings.filterwarnings('ignore')

SUBJECT_ID      = 'ZGW'
SUBJECT_VERSION = 'v1'

PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
ZUCO_DATA_PATH  = os.path.join(PROJECT_ROOT, 'dataset/zuco')
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, 'experiments/ZuCo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.beam_decoder import BeamDecoderIndoGPT
from model import ConformerIndoGPTTransducer as _   # path setup
from IndoGPT.decoder import IndoGPTDecoder
from conformer import Conformer
from joiner import JointNetwork

MODEL_PATH      = os.path.join(OUTPUT_DIR, 'ZuCo_ZGW_NR_logmel_GPT2frozen_model.pt')
OUTPUT_CSV_NAME = 'ZuCo_NOISE_BASELINE_ZGW_NR_logmel_GPT2frozen_test_predictions.csv'

EMOTIV_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
                   'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
EMOTIV_CHANNEL_INDICES = [2, 6, 4, 8, 14, 22, 26, 62, 58, 50, 42, 38, 40, 34]

CONFIG = {
    'input_dim':     14 * 64,
    'encoder_dim':   128,
    'decoder_dim':   768,
    'joint_dim':     768,
    'vocab_size':    None,
    'train_ratio':   0.70,
    'val_ratio':     0.10,
    'test_ratio':    0.20,
    'random_seed':   42,
    'sample_rate':   500,
    'n_fft':         256,
    'win_length':    256,
    'hop_length':    32,
    'n_mels':        64,
    'f_min':         0.5,
    'f_max':         45.0,
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,
    'encoder_dropout': 0.1,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConformerGPT2FrozenTransducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Conformer(
            input_dim=config['input_dim'],
            d_model=config['encoder_dim'],
            dropout=config.get('encoder_dropout', 0.1),
        )
        self.decoder = IndoGPTDecoder(
            model_name_or_path='gpt2',
            joint_dim=config['decoder_dim'],
        )
        self.joiner = JointNetwork(
            vocab_size=config['vocab_size'],
            encoder_dim=config['encoder_dim'],
            decoder_dim=config['decoder_dim'],
            joint_dim=config['joint_dim'],
        )

    def forward(self, audio_inputs, text_inputs):
        enc_out = self.encoder(audio_inputs)
        dec_out, _ = self.decoder(text_inputs)
        return self.joiner(enc_out, dec_out)

    def get_encoder_out_lengths(self, input_lengths):
        out = (input_lengths + 2 * 1 - 3) // 2 + 1
        return (out + 2 * 1 - 3) // 2 + 1

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[-–—]', ' ', text)
    text = re.sub(r"[^a-z0-9\s]", '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_zuco_sentences_scipy(filepath):
    mat  = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data = mat.get('sentenceData', mat.get('data', None))
    if data is None: raise KeyError(f"No 'sentenceData' in {filepath}")
    sentences, eeg_arrays = [], []
    for trial in (data if hasattr(data, '__iter__') else [data]):
        content = getattr(trial, 'content', None)
        if content is None: continue
        sentence = normalize_text(str(content))
        if not sentence: continue
        raw = getattr(trial, 'rawData', None)
        if raw is None or not isinstance(raw, np.ndarray) or raw.ndim != 2: continue
        if raw.shape[0] != 105: continue
        eeg_arrays.append(raw.T.astype(np.float32))
        sentences.append(sentence)
    return sentences, eeg_arrays

def load_zuco_sentences_h5py(filepath):
    import h5py
    sentences, eeg_arrays = [], []
    with h5py.File(filepath, 'r') as f:
        sd = f['sentenceData']
        for i in range(sd['content'].shape[0]):
            try:
                chars = f[sd['content'][i, 0]][()].flatten()
                sentence = normalize_text(''.join(chr(int(c)) for c in chars))
            except Exception: continue
            if not sentence: continue
            try: raw = f[sd['rawData'][i, 0]][()].astype(np.float32)
            except Exception: continue
            if raw.ndim != 2: continue
            if raw.shape[1] == 105: pass
            elif raw.shape[0] == 105: raw = raw.T
            else: continue
            eeg_arrays.append(raw); sentences.append(sentence)
    return sentences, eeg_arrays

def load_zuco_subject(subject_id, version):
    nr_dir   = os.path.join(ZUCO_DATA_PATH, version, 'NR')
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
    if not frontal_indices: return eeg_signal
    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try: components = ica.fit_transform(eeg_signal)
    except Exception: return eeg_signal
    bad = []
    for i in range(components.shape[1]):
        for f_idx in frontal_indices:
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold: bad.append(i); break
    if bad: components[:, bad] = 0.0
    return ica.inverse_transform(components)

def compute_logmel(eeg_signal, config):
    signal_tensor = torch.FloatTensor(eeg_signal.T)
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'], n_fft=config['n_fft'],
        win_length=config['win_length'], hop_length=config['hop_length'],
        f_min=config['f_min'], f_max=config['f_max'],
        n_mels=config['n_mels'], power=2.0,
    )
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    log_mel = db_transform(mel_transform(signal_tensor))
    log_mel_np = log_mel.numpy().transpose(2, 0, 1)
    features = log_mel_np.reshape(log_mel_np.shape[0], -1).astype(np.float32)
    return ((features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)).astype(np.float32)

def extract_features(eeg_arrays, sentences, indices, config):
    feats, tgts, meta = [], [], []
    for i in tqdm(indices, desc='  Extracting Log-Mel'):
        eeg = eeg_arrays[i][:, EMOTIV_CHANNEL_INDICES]
        if eeg.shape[0] < config['n_fft']: continue
        if config.get('remove_eye_artifacts', True):
            eeg = remove_ocular_artifacts_ica(eeg, EMOTIV_CHANNELS, config['ica_threshold'])
        feats.append(compute_logmel(eeg, config))
        tgts.append(sentences[i])
        meta.append({'sentence_idx': i, 'subject': SUBJECT_ID,
                     'version': SUBJECT_VERSION, 'sentence': sentences[i]})
    return feats, tgts, meta

def get_enc_out_len(feat_len):
    out = (feat_len - 1) // 2 + 1
    return (out - 1) // 2 + 1

def filter_split(feats, tgts, meta, tokenizer, batch_size=4, vocab_size=50258):
    INT32_MAX = 2 ** 31 - 1
    MAX_FEAT_LEN = 400
    kept_f, kept_t, kept_m = [], [], []
    dropped = 0
    for f, t, m in zip(feats, tgts, meta):
        enc_len = get_enc_out_len(f.shape[0])
        tok_len = len(tokenizer.encode(t))
        if tok_len == 0 or enc_len < tok_len or f.shape[0] > MAX_FEAT_LEN:
            dropped += 1; continue
        if batch_size * enc_len * (tok_len + 1) * vocab_size > INT32_MAX:
            dropped += 1; continue
        kept_f.append(f); kept_t.append(t); kept_m.append(m)
    if dropped:
        print(f'    Dropped {dropped} samples (filter_split)')
    return kept_f, kept_t, kept_m

class EEGDataset(Dataset):
    def __init__(self, features, targets, tokenizer, metadata=None):
        self.features = features; self.targets = targets
        self.tokenizer = tokenizer; self.metadata = metadata or [{}] * len(features)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx):
        shifted = [t + 1 for t in self.tokenizer.encode(self.targets[idx])]
        return {'feature': torch.FloatTensor(self.features[idx]),
                'target':  torch.LongTensor(shifted),
                'metadata': self.metadata[idx]}

def collate_batch(batch):
    features  = [item['feature'] for item in batch]
    targets   = [item['target']  for item in batch]
    max_f_len = max(f.shape[0] for f in features)
    max_t_len = max(len(t) for t in targets)
    return {
        'feature':        torch.stack([torch.nn.functional.pad(f, (0, 0, 0, max_f_len - f.shape[0])) for f in features]),
        'feature_length': torch.LongTensor([f.shape[0] for f in features]),
        'target':         torch.stack([torch.nn.functional.pad(t, (0, max_t_len - len(t))) for t in targets]),
        'target_length':  torch.LongTensor([len(t) for t in targets]),
        'metadata':       [item['metadata'] for item in batch],
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

def predict_on_noise(model, test_loader, device, beam_decoder):
    model.eval()
    rows = []
    print('\n' + '=' * 70)
    print('NOISE-BASELINE TEST (Gaussian noise — Jo et al. 2024)')
    print('=' * 70)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='  Noise inference'):
            real_features  = batch['feature'].to(device)
            noise_features = torch.randn_like(real_features)
            for i, meta in enumerate(batch['metadata']):
                gt   = meta['sentence']
                pred = beam_decoder.decode(noise_features[i:i+1])
                cer  = compute_cer(gt, pred)
                rows.append({'sentence': gt, 'prediction': pred, 'cer': cer,
                             'subject': meta['subject']})
    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    df.to_csv(csv_path, index=False)
    print(f'\n[SAVE] → {OUTPUT_CSV_NAME}')
    print(f'  Avg CER (noise): {df["cer"].mean():.4f}')
    return df

def main():
    print('=' * 70)
    print(f'ZuCo Noise-Baseline | ZGW | Log-Mel + GPT-2 Frozen (v13_1)')
    print(f'Device: {DEVICE}')
    print('=' * 70)

    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: Model not found: {MODEL_PATH}')
        return

    print('\n[STEP 1] Loading GPT2Tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    CONFIG['vocab_size'] = tokenizer.vocab_size + 1
    print(f'  Vocab size: {CONFIG["vocab_size"]}')

    print(f'\n[STEP 2] Loading ZuCo NR data — {SUBJECT_ID} ({SUBJECT_VERSION})...')
    sentences, eeg_arrays = load_zuco_subject(SUBJECT_ID, SUBJECT_VERSION)
    print(f'  {len(sentences)} sentences loaded')

    print(f'\n[STEP 3] Reproducing 70/10/20 split (seed={CONFIG["random_seed"]})...')
    idx_all = list(range(len(sentences)))
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=CONFIG['test_ratio'], random_state=CONFIG['random_seed'])
    print(f'  Test split: {len(idx_test)} samples')

    print('\n[STEP 4] Extracting Log-Mel features for test split...')
    feats_test, tgts_test, meta_test = extract_features(eeg_arrays, sentences, idx_test, CONFIG)

    feats_test, tgts_test, meta_test = filter_split(
        feats_test, tgts_test, meta_test, tokenizer,
        batch_size=4, vocab_size=CONFIG['vocab_size'])
    print(f'  Valid test samples after filter: {len(feats_test)}')

    test_dataset = EEGDataset(feats_test, tgts_test, tokenizer, meta_test)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

    print('\n[STEP 5] Loading model...')
    model = ConformerGPT2FrozenTransducer(CONFIG).to(DEVICE)
    saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved['model_state_dict'], strict=False)
    print(f'  Loaded: {os.path.basename(MODEL_PATH)}')

    beam_decoder = BeamDecoderIndoGPT(model, tokenizer, beam_size=3, max_sym_per_frame=15)
    predict_on_noise(model, test_loader, DEVICE, beam_decoder)

if __name__ == '__main__':
    main()
