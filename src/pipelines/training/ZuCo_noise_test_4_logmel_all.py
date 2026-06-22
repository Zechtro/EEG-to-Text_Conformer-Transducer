import os, sys, re, pickle, warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
import torchaudio.functional as AF
from sklearn.decomposition import FastICA
from scipy.stats import pearsonr
import scipy.io as sio
from sklearn.model_selection import train_test_split
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
ZUCO_DATA_PATH = os.path.join(PROJECT_ROOT, 'dataset/zuco')
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, 'experiments/ZuCo')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src/model'))
from misc.tokenizer import CharTokenizer
import misc.beam_decoder_char as beam_decoder_char
from model import ConformerTransducer

TOKENIZER_PATH  = os.path.join(OUTPUT_DIR, 'ZuCo_all_NR_log-mel_fast_char_tokenizer_6_1.pkl')
MODEL_PATH      = os.path.join(OUTPUT_DIR, 'ZuCo_all_NR_log-mel_fast_model_6_1.pt')
OUTPUT_CSV_NAME = 'ZuCo_NOISE_BASELINE_all_NR_log-mel_fast_test_predictions_6_1.csv'

ZUCO_V1_SUBJECTS = ['ZAB','ZDM','ZDN','ZGW','ZJM','ZJN','ZJS','ZKB','ZKH','ZKW','ZMG','ZPH']
ZUCO_V2_SUBJECTS = ['YAC','YAG','YAK','YDG','YDR','YFR','YFS','YHS','YIS','YLS','YMD','YMS',
                    'YRH','YRK','YRP','YSD','YSL','YTL']

EMOTIV_CHANNELS        = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
EMOTIV_CHANNEL_INDICES = [2,6,4,8,14,22,26,62,58,50,42,38,40,34]

CONFIG = {
    'input_dim':    14 * 64,
    'encoder_dim':  128,
    'decoder_dim':  128,
    'joint_dim':    128,
    'vocab_size':   None,
    'encoder_dropout': 0.2,
    'decoder_dropout': 0.2,
    'sample_rate':  500,
    'n_fft':        256,
    'win_length':   256,
    'hop_length':   32,
    'n_mels':       64,
    'f_min':        0.5,
    'f_max':        45.0,
    'train_ratio':  0.70,
    'val_ratio':    0.10,
    'test_ratio':   0.20,
    'random_seed':  42,
    'remove_eye_artifacts': True,
    'ica_threshold': 0.8,
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[-–—]', ' ', text)
    text = re.sub(r"[^a-z0-9\s]", '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_zuco_sentences_scipy(filepath):
    mat  = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    data = mat.get('sentenceData', mat.get('data', None))
    if data is None:
        raise KeyError(f"No 'sentenceData' in {filepath}")
    sentences, eeg_arrays = [], []
    for trial in (data if hasattr(data, '__iter__') else [data]):
        content  = getattr(trial, 'content', None)
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
                chars    = f[sd['content'][i, 0]][()].flatten()
                sentence = normalize_text(''.join(chr(int(c)) for c in chars))
            except Exception: continue
            if not sentence: continue
            try:
                raw = f[sd['rawData'][i, 0]][()].astype(np.float32)
            except Exception: continue
            if raw.ndim != 2: continue
            if raw.shape[1] == 105: pass
            elif raw.shape[0] == 105: raw = raw.T
            else: continue
            eeg_arrays.append(raw)
            sentences.append(sentence)
    return sentences, eeg_arrays

def load_zuco_subject(subject_id, version):
    nr_dir   = os.path.join(ZUCO_DATA_PATH, version, 'NR')
    filepath = os.path.join(nr_dir, f'results{subject_id}_NR.mat')
    if not os.path.isfile(filepath):
        return [], []
    try:
        return load_zuco_sentences_scipy(filepath)
    except NotImplementedError:
        return load_zuco_sentences_h5py(filepath)

def load_all_zuco_subjects():
    all_sentences, all_eeg, all_meta = [], [], []
    for subject_id, version in (
        [(sid, 'v1') for sid in ZUCO_V1_SUBJECTS] +
        [(sid, 'v2') for sid in ZUCO_V2_SUBJECTS]
    ):
        sents, eegs = load_zuco_subject(subject_id, version)
        if not sents:
            continue
        for i, (sent, eeg) in enumerate(zip(sents, eegs)):
            all_sentences.append(sent)
            all_eeg.append(eeg[:, EMOTIV_CHANNEL_INDICES])
            all_meta.append({'subject': subject_id, 'version': version,
                             'sentence_idx': i, 'sentence': sent})
    return all_sentences, all_eeg, all_meta

def remove_ocular_artifacts_ica(eeg_signal, ch_names, threshold=0.8):
    frontal = [i for i, ch in enumerate(ch_names) if 'AF3' in ch.upper() or 'AF4' in ch.upper()]
    if not frontal:
        return eeg_signal
    ica = FastICA(n_components=eeg_signal.shape[1], random_state=42, max_iter=1000, tol=0.01)
    try:
        components = ica.fit_transform(eeg_signal)
    except Exception:
        return eeg_signal
    bad = []
    for i in range(components.shape[1]):
        for f_idx in frontal:
            corr, _ = pearsonr(components[:, i], eeg_signal[:, f_idx])
            if abs(corr) > threshold:
                bad.append(i); break
    if bad:
        components[:, bad] = 0.0
    return ica.inverse_transform(components)

def extract_log_mel(eeg_signal, config):
    mel_transform = T.MelSpectrogram(
        sample_rate=config['sample_rate'],
        n_fft=config['n_fft'],
        win_length=config['win_length'],
        hop_length=config['hop_length'],
        n_mels=config['n_mels'],
        f_min=config['f_min'],
        f_max=config['f_max'],
    )
    channels = []
    for ch in range(eeg_signal.shape[1]):
        sig = torch.FloatTensor(eeg_signal[:, ch]).unsqueeze(0)
        mel = mel_transform(sig).squeeze(0)
        mel = AF.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10,
                                  db_multiplier=0.0, top_db=80.0)
        channels.append(mel)
    feat = torch.cat(channels, dim=0).T  # (T, 14*64)
    mean = feat.mean(dim=0, keepdim=True)
    std  = feat.std(dim=0, keepdim=True).clamp(min=1e-6)
    return ((feat - mean) / std).numpy().astype(np.float32)

class EEGDataset(Dataset):
    def __init__(self, features, targets, tokenizer, metadata=None):
        self.features  = features
        self.targets   = targets
        self.tokenizer = tokenizer
        self.metadata  = metadata or [{}] * len(features)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        return {
            'feature':  torch.FloatTensor(self.features[idx]),
            'target':   torch.LongTensor(self.tokenizer.text_to_int(self.targets[idx])),
            'metadata': self.metadata[idx],
        }

def collate_batch(batch):
    features = [item['feature'] for item in batch]
    targets  = [item['target']  for item in batch]
    max_f = max(f.shape[0] for f in features)
    max_t = max(len(t) for t in targets)
    return {
        'feature':        torch.stack([torch.nn.functional.pad(f,(0,0,0,max_f-f.shape[0])) for f in features]),
        'feature_length': torch.LongTensor([f.shape[0] for f in features]),
        'target':         torch.stack([torch.nn.functional.pad(t,(0,max_t-len(t))) for t in targets]),
        'target_length':  torch.LongTensor([len(t) for t in targets]),
        'metadata':       [item['metadata'] for item in batch],
    }

def compute_cer(ref, hyp):
    if not ref: return 1.0 if hyp else 0.0
    d = np.zeros((len(ref)+1, len(hyp)+1))
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            c = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+c)
    return d[len(ref)][len(hyp)] / len(ref)

def main():
    print('='*70)
    print('ZuCo Noise-Baseline | ALL Subjects | Log-Mel + Char LSTM (v6_1)')
    print(f'Device: {DEVICE}')
    print('='*70)

    print('\n[STEP 1] Loading CharTokenizer...')
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    CONFIG['vocab_size'] = tokenizer.vocab_size()
    print(f'  Vocab size: {CONFIG["vocab_size"]}')

    print('\n[STEP 2] Loading ZuCo NR data — all subjects (v1 + v2)...')
    all_sentences, all_eeg, all_meta = load_all_zuco_subjects()
    print(f'  Total sentences: {len(all_sentences)}')

    print(f'\n[STEP 3] Reproducing 70/10/20 split (seed={CONFIG["random_seed"]})...')
    idx = list(range(len(all_sentences)))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=CONFIG['test_ratio'], random_state=CONFIG['random_seed'])
    print(f'  Test split: {len(idx_test)} samples')

    print('\n[STEP 4] Extracting Log-Mel features for test split...')
    feats, tgts, meta = [], [], []
    for i in tqdm(idx_test, desc='  Extracting Log-Mel'):
        eeg = all_eeg[i]
        if eeg.shape[0] < CONFIG['n_fft']:
            continue
        if CONFIG['remove_eye_artifacts']:
            eeg = remove_ocular_artifacts_ica(eeg, EMOTIV_CHANNELS, CONFIG['ica_threshold'])
        feats.append(extract_log_mel(eeg, CONFIG))
        tgts.append(all_sentences[i])
        meta.append(all_meta[i])
    print(f'  Valid test samples: {len(feats)}')

    print('\n[STEP 5] Loading model...')
    model = ConformerTransducer(CONFIG).to(DEVICE)
    saved = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(saved['model_state_dict'], strict=False)
    print(f'  Loaded: {os.path.basename(MODEL_PATH)}')

    test_ds     = EEGDataset(feats, tgts, tokenizer, meta)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_batch)
    beam_dec    = beam_decoder_char.BeamDecoderChar(model, tokenizer, beam_size=3, max_sym_per_frame=15)

    model.eval()
    rows = []
    print('\n' + '='*70)
    print('NOISE-BASELINE TEST (Gaussian noise — Jo et al. 2024)')
    print('='*70)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='  Noise inference'):
            noise = torch.randn_like(batch['feature'].to(DEVICE))
            for i, m in enumerate(batch['metadata']):
                gt   = m['sentence']
                pred = beam_dec.decode(noise[i:i+1])
                rows.append({'sentence': gt, 'prediction': pred,
                             'cer': compute_cer(gt, pred),
                             'subject': m['subject'], 'version': m['version']})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV_NAME)
    df.to_csv(csv_path, index=False)
    print(f'\n[SAVE] → {OUTPUT_CSV_NAME}')
    print(f'  Avg CER (noise): {df["cer"].mean():.4f}')

if __name__ == '__main__':
    main()
