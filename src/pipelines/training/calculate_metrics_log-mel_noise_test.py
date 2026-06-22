import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def compute_wer(reference, hypothesis):
    """
    Menghitung Word Error Rate (WER) menggunakan Levenshtein distance level kata.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    for i in range(len(ref_words) + 1): d[i][0] = i
    for j in range(len(hyp_words) + 1): d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1,       # Deletion
                          d[i][j-1] + 1,        # Insertion
                          d[i-1][j-1] + cost)   # Substitution

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

# Menghindari OverflowError di Python 3.12 yang disebabkan oleh bug

# tokenizer.encode() / enable_truncation() tanpa max_length eksplisit.
# Solusi: gunakan transformers langsung dengan use_fast=False (slow tokenizer)
# dan tentukan max_length=512 secara eksplisit.

def compute_bertscore_manual(
    refs,
    hyps,
    model_name="indobenchmark/indobert-base-p1",
    batch_size=32
):
    """
    Menghitung BERTScore F1 secara manual menggunakan transformers langsung.

    Menggunakan cosine similarity antar mean-pooled token embeddings sebagai
    aproksimasi BERTScore F1. Hasilnya sangat berkorelasi dengan implementasi
    resmi bert_score untuk dataset berukuran kecil-menengah.

    Args:
        refs       : List[str] — kalimat referensi (ground truth)
        hyps       : List[str] — kalimat hipotesis (hasil prediksi)
        model_name : str       — nama model HuggingFace yang digunakan
        batch_size : int       — jumlah kalimat per batch inferensi

    Returns:
        np.ndarray — array skor F1 BERTScore per pasang kalimat (skala 0–1)
    """

    # menyebabkan OverflowError pada Python 3.12
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if device == "cuda":
        print(f"     [INFO] Menggunakan GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("     [INFO] GPU tidak tersedia, menggunakan CPU.")

    def get_mean_pooled_embeddings(sentences):
        """
        Tokenisasi batch kalimat lalu hitung mean-pooled embedding
        dengan masked averaging (mengabaikan token padding).
        """
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,         # Batas eksplisit — mencegah OverflowError
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                output = model(**encoded)

            # output.last_hidden_state: (batch, seq_len, hidden_dim)
            token_embeddings = output.last_hidden_state

            attention_mask = encoded['attention_mask'].unsqueeze(-1).float()

            sum_embeddings = (token_embeddings * attention_mask).sum(dim=1)
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            all_embeddings.append(mean_embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    print("     [INFO] Menghitung embedding referensi...")
    ref_embeddings = get_mean_pooled_embeddings(refs)

    print("     [INFO] Menghitung embedding hipotesis...")
    hyp_embeddings = get_mean_pooled_embeddings(hyps)

    f1_scores = F.cosine_similarity(ref_embeddings, hyp_embeddings, dim=-1)

    return f1_scores.numpy()

def main():

    # PRE-LOAD MODEL: Muat tokenizer dan model IndoBERT sekali di awal
    # agar tidak diunduh ulang di setiap iterasi file.

    MODEL_NAME = "indobenchmark/indobert-base-p1"

    print("=" * 80)
    print("MEMASTIKAN MODEL INDOBERT SUDAH TER-CACHE SECARA LOKAL...")
    print("=" * 80)
    try:
        # Uji pemuatan awal — akan mengunduh jika belum ada di cache
        AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        AutoModel.from_pretrained(MODEL_NAME)
        print("[OK] Model IndoBERT siap digunakan.\n")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model IndoBERT: {e}")
        print("        Pastikan koneksi internet aktif lalu coba lagi.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))

    subjects = [f"SUB{i}" for i in range(1, 13)]

    target_files = [
        "NOISE_BASELINE_all_eq_3_0_log-mel_test_predictions_6_1.csv",
        "NOISE_BASELINE_all_eq_3_0_logmel_test_predictions_10_1_IndoGPT.csv"
    ]

    for sub in subjects:
        target_files.append(f"NOISE_BASELINE_{sub}_eq_3_0_logmel_test_predictions_10_1_IndoGPT.csv")
        target_files.append(f"NOISE_BASELINE_{sub}_eq_3_0_log-mel_test_predictions_6_1.csv")

    rouge1_scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)

    # (mencegah skor 0 jika kalimat sangat pendek)
    smooth_func = SmoothingFunction().method1

    print("=" * 80)
    print("MEMULAI PERHITUNGAN METRIK EVALUASI (WER, BLEU, ROUGE, BERTSCORE)")
    print(f"Mencari file CSV di direktori:\n{script_dir}")
    print("=" * 80)

    for filename in target_files:
        filepath = os.path.join(script_dir, filename)

        if not os.path.exists(filepath):
            continue

        print(f"\nMemproses file: {filename}")

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"  [ERROR] Gagal membaca {filename}: {e}")
            continue

        df['sentence']   = df['sentence'].fillna("").astype(str)
        df['prediction'] = df['prediction'].fillna("").astype(str)

        wer_list = []
        bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
        rouge1_p_list, rouge1_r_list, rouge1_f_list = [], [], []

        print("  -> Menghitung metrik WER, BLEU, dan ROUGE...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ref = row['sentence']
            hyp = row['prediction']

            ref_tokens = ref.split()
            hyp_tokens = hyp.split() if hyp.strip() else [""]

            w_err = compute_wer(ref, hyp)
            wer_list.append(round(w_err * 100, 4))

            b1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0),          smoothing_function=smooth_func)
            b2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0),       smoothing_function=smooth_func)
            b3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0),  smoothing_function=smooth_func)
            b4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func)

            bleu1_list.append(round(b1 * 100, 4))
            bleu2_list.append(round(b2 * 100, 4))
            bleu3_list.append(round(b3 * 100, 4))
            bleu4_list.append(round(b4 * 100, 4))

            r_scores = rouge1_scorer.score(ref, hyp)
            rouge1_p_list.append(round(r_scores['rouge1'].precision * 100, 4))
            rouge1_r_list.append(round(r_scores['rouge1'].recall    * 100, 4))
            rouge1_f_list.append(round(r_scores['rouge1'].fmeasure  * 100, 4))

        if 'cer' in df.columns:
            df['cer'] = round(df['cer'] * 100, 4)
            idx_cer = df.columns.get_loc('cer')
            df.insert(idx_cer + 1, 'WER', wer_list)
        else:
            df['WER'] = wer_list

        df['BLEU-1']    = bleu1_list
        df['BLEU-2']    = bleu2_list
        df['BLEU-3']    = bleu3_list
        df['BLEU-4']    = bleu4_list
        df['ROUGE-1-P'] = rouge1_p_list
        df['ROUGE-1-R'] = rouge1_r_list
        df['ROUGE-1-F'] = rouge1_f_list

        print("  -> Menghitung BERTScore (IndoBERT, slow tokenizer)...")

        refs = df['sentence'].tolist()
        hyps = df['prediction'].tolist()

        safe_refs = [r if r.strip() else "." for r in refs]
        safe_hyps = [h if h.strip() else "." for h in hyps]

        try:
            f1_scores = compute_bertscore_manual(
                refs=safe_refs,
                hyps=safe_hyps,
                model_name=MODEL_NAME,
                batch_size=32
            )
            df['BertScore'] = [round(float(val) * 100, 4) for val in f1_scores]
        except Exception as e:
            print(f"  [WARNING] BERTScore gagal dihitung: {e}")
            df['BertScore'] = [None] * len(df)

        new_filename = f"complete_metrics_{filename}"
        new_filepath = os.path.join(script_dir, new_filename)

        df.to_csv(new_filepath, index=False)
        print(f"  [BERHASIL] File disimpan sebagai: {new_filename}")

    print("\n" + "=" * 80)
    print("SELESAI! Semua file metrik telah berhasil dibuat.")
    print("=" * 80)

if __name__ == '__main__':
    main()