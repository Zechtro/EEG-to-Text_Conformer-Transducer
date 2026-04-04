# QUICK START GUIDE - all0.py

## 🚀 Mulai dalam 3 Langkah

### 1️⃣ Setup
```bash
cd /Users/steven/Documents/Github\ Repositories/EEG-to-Text_Conformer-Transducer
source venv_conf/bin/activate
```

### 2️⃣ Run
```bash
python src/pipelines/training/all0.py
```

### 3️⃣ Hasilkan
File output akan tersembunyisimpen di `src/pipelines/training/`:
- `test_results.csv` - Hasil prediksi
- `model.pt` - Model terlatih
- `training_history.json` - Loss & CER history

---

## 📋 Checklist Sebelum Menjalankan

- [x] Venv activated: `source venv_conf/bin/activate`
- [x] File ada: `/dataset/cleaned_transcript_mapping.csv`
- [x] Folder ada: `/dataset/raw/`
- [x] PyTorch installed: `python -c "import torch; print(torch.__version__)"`

---

## ⚙️ Kustomisasi (opsional)

Edit baris 45-70 di `all0.py`:

```python
CONFIG = {
    'batch_size': 8,        # ← Kurangi jika memory error
    'num_epochs': 50,       # ← Tambah untuk performa lebih baik
    'learning_rate': 1e-3,  # ← Try 1e-4 jika overfitting
}
```

---

## 📊 Cek Hasil

Buka file hasil:
```bash
# Lihat test predictions
cat src/pipelines/training/test_results.csv | head -5

# Cek training history
cat src/pipelines/training/training_history.json | python -m json.tool
```

---

## 🐛 Debug

**Training macet?**
- Reduce batch_size: `'batch_size': 4`

**Memory error?**
- Reduce n_mels: `'n_mels': 40` (dari 80)

**Hasil jelek (CER tinggi)?**
- Increase epochs: `'num_epochs': 100`
- Lower learning rate: `'learning_rate': 1e-4`

---

## 📈 Expected Results

Typical metrics setelah training:
- **Final Val CER:** 0.15-0.35 (tergantung data complexity)
- **Training time:** 1-3 jam (CPU/GPU)
- **Model size:** ~10-50 MB

---

## 📚 Dokumentasi Lengkap

- `README.md` - Detailed documentation
- `IMPLEMENTATION_SUMMARY.md` - Feature breakdown
- `all0.py` - Fully commented source code

---

**Questions?** Check docstrings dalam `all0.py` atau baca README.md
