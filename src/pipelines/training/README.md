# EEG-to-Text Training Pipeline

File: `all0.py`

## Deskripsi
Pipeline training lengkap untuk model Conformer-Transducer dengan input EEG signals dan output text transcriptions.

## Fitur Utama

1. **Data Loading & Preprocessing**
   - Load dataset CSV dari `cleaned_transcript_mapping.csv`
   - Load EEG signals dari file CSV mentah (14 channel: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
   - Data sudah di-bandpass 50Hz dari EmotivPro

2. **Dataset Split**
   - Train: 70% (dengan constraint: kalimat yang sama harus ada di bagian yang sama)
   - Validation: 10%
   - Test: 20%

3. **Feature Extraction**
   - Log Mel Spectrogram dari setiap channel EEG
   - Sampling rate: 256 Hz
   - n_mels: 80
   - Output shape per sample: (time_steps, 14*80=1120)

4. **Model Architecture**
   - Encoder: Conformer (encoder_dim=256)
   - Decoder: LSTM (decoder_dim=512)
   - Joiner: Neural network layer (joint_dim=512)
   - Vocab size: Character-level (otomatis dari training set)

5. **Training**
   - Optimizer: Adam (lr=1e-3)
   - Batch size: 8
   - Epochs: 50
   - Loss: CrossEntropy
   - GPU CUDA support dengan fallback ke CPU

6. **Evaluation Metrics**
   - Character Error Rate (CER) menggunakan edit distance
   - Tracked untuk train dan validation sets

7. **Output Files**
   - `tokenizer.json` - Character vocabulary and mapping
   - `model.pt` - Trained model weights
   - `training_history.json` - Training/validation loss and CER per epoch
   - `test_results.csv` - Predictions on test set dengan format:
     - id: recording ID
     - subject: subject name
     - gender: male/female
     - target_sentence: original sentence
     - predicted_sentence: model prediction
     - cer: character error rate untuk record tersebut

## Cara Menjalankan

### Prerequisites
```bash
# Pastikan sudah di dalam venv
source venv_conf/bin/activate

# Install required packages jika belum
pip install torch torchaudio librosa pandas numpy scikit-learn
```

### Run Training
```bash
cd /Users/steven/Documents/Github\ Repositories/EEG-to-Text_Conformer-Transducer

# Jalankan pipeline
python src/pipelines/training/all0.py
```

### Expected Output
```
================================================================================
EEG-to-Text Conformer-Transducer Training Pipeline
================================================================================

[STEP 1] Load dataset CSV...
Total records: 1050
[STEP 2] Split dataset (70% train, 10% val, 20% test)...
[STEP 3] Load & process EEG signals, compute Log Mel Spectrograms...
[STEP 4] Build Character Tokenizer...
Vocab size: 35
[STEP 5] Create PyTorch Datasets...
[STEP 5b] Build model...
[STEP 6] Training model...
[Epoch 1/50]
Train Loss: X.XXXX
Val Loss: X.XXXX | Val CER: X.XXXX
...
[STEP 7+8] Predicting on test set & computing CER...
[STEP 9] Save results...
[SAVE] Results saved to src/pipelines/training/test_results.csv
[SUMMARY] Average Test CER: X.XXXX

================================================================================
Training completed!
================================================================================
```

### Hasil Training
Setelah selesai, file-file berikut akan tersedia di `src/pipelines/training/`:
- `training_history.json` - Loss dan CER history per epoch
- `model.pt` - Model yang sudah dilatih
- `tokenizer.json` - Character tokenizer  
- `test_results.csv` - Hasil prediksi pada test set

### Analisis Hasil
Buka file `test_results.csv` untuk melihat:
```
id,subject,gender,target_sentence,predicted_sentence,cer
1_DAM,DAM,male,siapa wanita itu,siapa wanita itu,0.0
2_DAM,DAM,male,anda benar benar kakek yang murah hati,anda benar kakek yang murah hati,0.05
...
```

## Konfigurasi
Edit `CONFIG` dictionary di dalam `all0.py` untuk mengubah:
- `batch_size`: Ukuran batch (default: 8)
- `num_epochs`: Jumlah epoch training (default: 50)
- `learning_rate`: Learning rate (default: 1e-3)
- `train_ratio`, `val_ratio`, `test_ratio`: Proporsi dataset split

## Notes
- GPU CUDA akan digunakan otomatis jika tersedia (`torch.cuda.is_available()`)
- Jika tidak ada GPU, akan fallback ke CPU
- Training time tergantung:
  - Jumlah samples yang berhasil di-load
  - Ukuran GPU/CPU
  - Jumlah epochs
  
## Troubleshooting

### Error: "CSV file not found"
- Pastikan path `/dataset/cleaned_transcript_mapping.csv` sudah benar
- Pastikan struktur folder `/dataset/raw/` sudah dengan format `gender/subject/csv/`

### Error: "Memory overflow"
- Kurangi `batch_size` di CONFIG (contoh: 4 atau 2)
- Kurangi `n_mels` dari 80 menjadi 64 atau 40

### Prediksi kualitas buruk (CER tinggi)
- Increase `num_epochs`
- Try different `learning_rate` (1e-4 hingga 1e-2)
- Check if training data preprocessing is correct
