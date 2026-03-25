# TODO
import torch
import torchaudio
import pandas as pd
import os
import librosa
import re
from torch.utils.data import Dataset

try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

class JavaneseDataset(Dataset):
    def __init__(self, csv_file, audio_root, tokenizer, augment=False, trim_silence=False):
        self.df = pd.read_csv(csv_file)
        self.audio_root = audio_root
        self.tokenizer = tokenizer
        
        # Gunakan Cleaned_Transcript jika ada, jika tidak gunakan Transcript
        if 'Cleaned_Transcript' in self.df.columns:
            print(f"[Dataset] Menggunakan Cleaned_Transcript dari file: {csv_file}")
            self.transcript_col = 'Cleaned_Transcript'
        else:
            print(f"[Dataset] Cleaned_Transcript tidak ditemukan, menggunakan Transcript")
            self.transcript_col = 'Transcript'
        
        # Config
        self.augment = augment
        self.trim_silence = trim_silence
        self.sample_rate = 16000
        
        # MelSpectogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_mels=80, n_fft=400, hop_length=160
        )
        
        # SpecAugment
        self.spec_augment = torch.nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
            torchaudio.transforms.TimeMasking(time_mask_param=10)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        audio_path = os.path.join(self.audio_root, f"{row['SentenceID']}.wav")
        if not os.path.exists(audio_path):
            audio_path = self._find_audio_case_insensitive(row['SentenceID'])
            if not audio_path:
                return None, None

        # Gunakan kolom transcript yang sudah ditentukan di __init__
        transcript = row[self.transcript_col]
        
        # Load Audio
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            # Stereo -> Mono
            if waveform.shape[0] > 1: waveform = waveform[0:1, :] 
            
            # Resample
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            
            # Trim Silence
            if self.trim_silence:
                wav_np = waveform.squeeze().numpy()
                trimmed, _ = librosa.effects.trim(wav_np, top_db=20)
                if len(trimmed) > (0.1 * self.sample_rate):
                    waveform = torch.from_numpy(trimmed).unsqueeze(0)

            # Mel Spectrogram
            spec = self.mel_transform(waveform)
            
            # Log-Mel
            spec = torch.log(spec + 1e-9)
            
            # SpecAugment
            if self.augment:
                spec = self.spec_augment(spec)
                
            spec = spec.squeeze(0).transpose(0, 1) 
            targets = torch.tensor(self.tokenizer.text_to_int(transcript), dtype=torch.long)
            return spec, targets
            
        except Exception as e:
            print(f"[Dataset Error] File: {audio_path} | {e}")
            return None, None
    
    def _find_audio_case_insensitive(self, sentence_id):
        
        filename = f"{sentence_id}.wav"
        
        def normalize_id(id_str):
            
            # Mengubah inkonsistensi penomoran seperti utt09 menjadi utt9
            return re.sub(r'(\D)0+(\d+)', r'\1\2', id_str)
        
        # Search in main directory
        if os.path.exists(self.audio_root):
            try:
                files = os.listdir(self.audio_root)
                filename_lower = filename.lower()
                sentence_id_lower = sentence_id.lower()
                sentence_id_normalized = normalize_id(sentence_id_lower)
                
                for file in files:
                    if not file.lower().endswith('.wav'):
                        continue
                    
                    if file.lower() == filename_lower:
                        return os.path.join(self.audio_root, file)
                    
                    file_without_ext = file[:-4]
                    file_normalized = normalize_id(file_without_ext.lower())
                    
                    if file_normalized == sentence_id_normalized:
                        return os.path.join(self.audio_root, file)
            except Exception:
                pass
        return None

def collate_fn(batch):
    # Filter error/None
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, None, None, None
    
    specs, targets = zip(*batch)
    input_lengths = torch.tensor([s.size(0) for s in specs], dtype=torch.long)
    padded_specs = torch.nn.utils.rnn.pad_sequence(specs, batch_first=True)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
    return padded_specs, padded_targets, input_lengths, target_lengths