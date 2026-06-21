import os
import pandas as pd
from pathlib import Path

# ============================================================================
# KONFIGURASI PATH & MAPPING
# ============================================================================
# Menggunakan absolute path sesuai direktori Anda
BASE_DIR = Path(r"D:\GitRepos\EEG-to-Text_Conformer-Transducer\dataset")
RAW_DIR = BASE_DIR / "raw"
CLEANED_CSV = BASE_DIR / "cleaned_transcript_mapping.csv"
RAW_CSV = BASE_DIR / "raw_transcript_mapping.csv"

# Dictionary Mapping Subjek
# Silakan tambahkan atau ubah subjek lain di dalam dictionary ini
SUBJECT_MAPPING = {
    "DAM": "SUB2",
    "RAN": "SUB3",
    "MAR": "SUB4",
    "NAU": "SUB1",
    "FAR": "SUB7",
    "EVE": "SUB9",
    "KEN": "SUB10",
    "RET": "SUB12",
    "ERI": "SUB5",
    "BEL": "SUB11",
    "LIA": "SUB6",
    "SUL": "SUB8"
    # "KODE_LAMA": "KODE_BARU",
}

# ============================================================================
# FUNGSI 1: UPDATE ISI FILE CSV
# ============================================================================
def update_csv_files():
    print("--- 1. Memperbarui File CSV ---")
    csv_files = [CLEANED_CSV, RAW_CSV]
    
    for csv_path in csv_files:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            # Update nilai di kolom 'subject' jika ada (Misal: "DAM" menjadi "SUB1")
            if 'subject' in df.columns:
                df['subject'] = df['subject'].replace(SUBJECT_MAPPING)
            
            # Update nilai di kolom 'id' jika ada (Misal: "1_DAM" menjadi "1_SUB1")
            if 'id' in df.columns:
                for old_sub, new_sub in SUBJECT_MAPPING.items():
                    # str.replace akan mencari substring dan menggantinya
                    df['id'] = df['id'].str.replace(old_sub, new_sub)
            
            # Simpan kembali ke CSV
            df.to_csv(csv_path, index=False)
            print(f"[\u2713] Berhasil memperbarui data di: {csv_path.name}")
        else:
            print(f"[\u26a0\ufe0f] File tidak ditemukan: {csv_path.name}")

# ============================================================================
# FUNGSI 2: UPDATE NAMA FILE DAN FOLDER DI RAW DATA
# ============================================================================
def rename_raw_files_and_folders():
    print("\n--- 2. Mengubah Nama File dan Folder di raw/ ---")
    if not RAW_DIR.exists():
        print(f"[\u26a0\ufe0f] Folder raw tidak ditemukan di {RAW_DIR}")
        return

    # topdown=False SANGAT PENTING:
    # Memaksa Python membaca dari direktori terdalam (file) ke terluar (folder).
    # Jika folder diubah duluan, path ke file di dalamnya akan error/not found.
    for root, dirs, files in os.walk(RAW_DIR, topdown=False):
        root_path = Path(root)
        
        # 1. Ubah nama file-file di dalamnya (misal '1_DAM.bdf' -> '1_SUB1.bdf')
        for file_name in files:
            new_name = file_name
            for old_sub, new_sub in SUBJECT_MAPPING.items():
                if old_sub in new_name:
                    new_name = new_name.replace(old_sub, new_sub)
            
            if new_name != file_name:
                old_file_path = root_path / file_name
                new_file_path = root_path / new_name
                old_file_path.rename(new_file_path)
                print(f"  \u251c\u2500 [File]   {file_name}  ->  {new_name}")

        # 2. Ubah nama folder (misal folder 'DAM' -> 'SUB1')
        for dir_name in dirs:
            new_dir_name = dir_name
            for old_sub, new_sub in SUBJECT_MAPPING.items():
                # Pastikan hanya mengubah folder yang namanya SAMA PERSIS dengan kode lama
                if dir_name == old_sub:
                    new_dir_name = new_sub
                    break
            
            if new_dir_name != dir_name:
                old_dir_path = root_path / dir_name
                new_dir_path = root_path / new_dir_name
                old_dir_path.rename(new_dir_path)
                print(f"  \u2514\u2500 [Folder] {dir_name}  ->  {new_dir_name}")

# ============================================================================
# MAIN EXECUTOR
# ============================================================================
if __name__ == "__main__":
    print("=====================================================")
    print(" MEMULAI PROSES REFACTORING NAMA SUBJEK")
    print("=====================================================\n")
    
    update_csv_files()
    rename_raw_files_and_folders()
    
    print("\n=====================================================")
    print(" \u2728 PROSES REFACTORING SELESAI \u2728")
    print("=====================================================")