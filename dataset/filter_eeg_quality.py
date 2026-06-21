import os
import pandas as pd
import glob
from tqdm import tqdm

# Konstanta Jalur File
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_CSV_PATH = os.path.join(BASE_DIR, 'cleaned_transcript_mapping.csv')
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, 'cleaned_transcript_mapping_eq_3_5.csv')
RAW_DIR = os.path.join(BASE_DIR, 'raw')
THRESHOLD_EQ = 3.5

# Daftar kolom EQ sesuai format dari header file Anda
EQ_COLUMNS = [
    'EQ.AF3', 'EQ.F7', 'EQ.F3', 'EQ.FC5', 'EQ.T7', 
    'EQ.P7', 'EQ.O1', 'EQ.O2', 'EQ.P8', 'EQ.T8', 
    'EQ.FC6', 'EQ.F4', 'EQ.F8', 'EQ.AF4'
]

def get_eeg_file_path(gender, subject, file_id):
    """Mencari jalur file .bp.csv berdasarkan parameter dari mapping"""
    search_pattern = os.path.join(RAW_DIR, gender, subject, 'csv', f"{file_id}_*.bp.csv")
    matching_files = glob.glob(search_pattern)
    
    if matching_files:
        return matching_files[0]
    return None

def calculate_average_eq(file_path):
    """
    Membaca data EQ menggunakan Pandas dengan melewati baris pertama (metadata).
    Menghitung rata-rata EQ dari seluruh baris data yang ada.
    """
    try:
        # skiprows=1: Abaikan metadata di baris paling atas
        # usecols: Hanya muat 14 kolom EQ ke memori agar proses sangat cepat
        df_eq = pd.read_csv(
            file_path, 
            skiprows=1, 
            usecols=lambda c: c in EQ_COLUMNS,
            low_memory=False
        )
        
        # Hitung rata-rata tiap kolom, lalu rata-ratakan semuanya
        # Otomatis mengabaikan nilai kosong (NaN)
        overall_avg = df_eq.mean().mean()
        
        # Jika semua nilai kosong (NaN), kembalikan 0.0
        if pd.isna(overall_avg):
            return 0.0
            
        return overall_avg
        
    except Exception as e:
        print(f"\n[ERROR] Gagal memproses EQ dari {os.path.basename(file_path)}: {e}")
        return 0.0

def main():
    print(f"Membuka file mapping: {MAPPING_CSV_PATH}")
    
    if not os.path.exists(MAPPING_CSV_PATH):
        print("Error: File cleaned_transcript_mapping.csv tidak ditemukan!")
        return

    df_mapping = pd.read_csv(MAPPING_CSV_PATH)
    filtered_rows = []

    print("Menganalisis rata-rata EQ untuk setiap rekaman...")
    
    for index, row in tqdm(df_mapping.iterrows(), total=len(df_mapping)):
        file_id = str(row['id'])
        subject = str(row['subject'])
        gender = str(row['gender'])
        
        eeg_file_path = get_eeg_file_path(gender, subject, file_id)
        
        if eeg_file_path is None:
            continue
            
        avg_eq = calculate_average_eq(eeg_file_path)
        
        # Filter jika rata-rata keseluruhan rekaman >= THRESHOLD_EQ
        if round(avg_eq, 1) >= THRESHOLD_EQ:
            filtered_rows.append(row)

    if filtered_rows:
        df_filtered = pd.DataFrame(filtered_rows)
        df_filtered.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\nSelesai! Ditemukan {len(filtered_rows)} dari {len(df_mapping)} rekaman dengan rata-rata EQ >= {THRESHOLD_EQ}")
        print(f"File tersimpan di: {OUTPUT_CSV_PATH}")
    else:
        print(f"\nSelesai! Tidak ada rekaman dengan rata-rata EQ >= {THRESHOLD_EQ} yang ditemukan.")

if __name__ == "__main__":
    main()