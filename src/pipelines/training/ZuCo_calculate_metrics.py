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
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def compute_bertscore_manual(refs, hyps, model_name='bert-base-uncased', batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModel.from_pretrained(model_name)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f'     [INFO] BERTScore device: {device}')

    def embed(sentences):
        all_emb = []
        for i in range(0, len(sentences), batch_size):
            batch   = sentences[i:i + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=512, return_tensors='pt').to(device)
            with torch.no_grad():
                out = model(**encoded)
            token_emb  = out.last_hidden_state
            mask       = encoded['attention_mask'].unsqueeze(-1).float()
            sum_emb    = (token_emb * mask).sum(dim=1)
            sum_mask   = mask.sum(dim=1).clamp(min=1e-9)
            all_emb.append((sum_emb / sum_mask).cpu())
        return torch.cat(all_emb, dim=0)

    print('     [INFO] Embedding referensi...')
    ref_emb = embed(refs)
    print('     [INFO] Embedding hipotesis...')
    hyp_emb = embed(hyps)
    return F.cosine_similarity(ref_emb, hyp_emb, dim=-1).numpy()

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'experiments/ZuCo')
    MODEL_NAME   = 'bert-base-uncased'

    TARGET_FILES = [
        'ZuCo_ZGW_NR_log-mel_fast_test_predictions_6_1.csv',
        'ZuCo_ZGW_NR_hilbert_fast_test_predictions_6_1.csv',
        'ZuCo_ZGW_NR_hilbert_GPT2_fast_test_predictions_10_1.csv',
        'ZuCo_ZGW_NR_logmel_GPT2frozen_test_predictions.csv',
    ]

    print('=' * 80)
    print('ZuCo — PERHITUNGAN METRIK (WER, BLEU, ROUGE, BERTScore)')
    print(f'Direktori: {OUTPUT_DIR}')
    print('=' * 80)

    print('\nMemastikan model IndoBERT sudah ter-cache...')
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        AutoModel.from_pretrained(MODEL_NAME)
        print('[OK] IndoBERT siap.\n')
    except Exception as e:
        print(f'[ERROR] Gagal memuat IndoBERT: {e}')
        return

    rouge1_scorer_obj = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=False)
    smooth_func = SmoothingFunction().method1

    for filename in TARGET_FILES:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f'[SKIP] Tidak ditemukan: {filename}')
            continue

        print(f'\n{"="*60}')
        print(f'Memproses: {filename}')
        print('='*60)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f'  [ERROR] {e}')
            continue

        df['sentence']   = df['sentence'].fillna('').astype(str)
        df['prediction'] = df['prediction'].fillna('').astype(str)

        wer_list = []
        bleu1_list, bleu2_list, bleu3_list, bleu4_list = [], [], [], []
        rouge1_p_list, rouge1_r_list, rouge1_f_list = [], [], []

        print('  -> WER, BLEU, ROUGE...')
        for _, row in tqdm(df.iterrows(), total=len(df)):
            ref = row['sentence']
            hyp = row['prediction']
            ref_tok = ref.split()
            hyp_tok = hyp.split() if hyp.strip() else ['']

            wer_list.append(round(compute_wer(ref, hyp) * 100, 4))

            bleu1_list.append(round(sentence_bleu([ref_tok], hyp_tok, weights=(1, 0, 0, 0),            smoothing_function=smooth_func) * 100, 4))
            bleu2_list.append(round(sentence_bleu([ref_tok], hyp_tok, weights=(0.5, 0.5, 0, 0),         smoothing_function=smooth_func) * 100, 4))
            bleu3_list.append(round(sentence_bleu([ref_tok], hyp_tok, weights=(0.33, 0.33, 0.33, 0),    smoothing_function=smooth_func) * 100, 4))
            bleu4_list.append(round(sentence_bleu([ref_tok], hyp_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_func) * 100, 4))

            r = rouge1_scorer_obj.score(ref, hyp)
            rouge1_p_list.append(round(r['rouge1'].precision * 100, 4))
            rouge1_r_list.append(round(r['rouge1'].recall    * 100, 4))
            rouge1_f_list.append(round(r['rouge1'].fmeasure  * 100, 4))

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

        print('  -> BERTScore (IndoBERT)...')
        safe_refs = [r if r.strip() else '.' for r in df['sentence'].tolist()]
        safe_hyps = [h if h.strip() else '.' for h in df['prediction'].tolist()]
        try:
            f1_scores = compute_bertscore_manual(safe_refs, safe_hyps, MODEL_NAME)
            df['BertScore'] = [round(float(v) * 100, 4) for v in f1_scores]
        except Exception as e:
            print(f'  [WARNING] BERTScore gagal: {e}')
            df['BertScore'] = [None] * len(df)

        new_filename = f'complete_metrics_{filename}'
        df.to_csv(os.path.join(OUTPUT_DIR, new_filename), index=False)
        print(f'  [SAVED] {new_filename}')

    print('\n' + '=' * 80)
    print('SELESAI.')
    print('=' * 80)

if __name__ == '__main__':
    main()
