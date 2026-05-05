import os
import pandas as pd

def main():
    # Mengambil lokasi absolut dari folder tempat script ini (evaluate_cer.py) berada
    # Ini memastikan script selalu mencari di folder yang benar, dari mana pun Anda memanggilnya (root/dll)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 65)
    print("REKAP EVALUASI RATA-RATA CER EKSMPERIMEN HILBERT")
    print("=" * 65)
    
    # Loop dari versi 2 hingga 8
    for i in range(2, 9):
        # Format nama file sesuai pola
        filename = f"SUB1_hilbert_test_predictions_{i}_0.csv"
        # Gabungkan path folder script dengan nama file
        filepath = os.path.join(script_dir, filename)
        
        # Cek apakah file benar-benar ada di komputer
        if os.path.exists(filepath):
            try:
                # Baca CSV
                df = pd.read_csv(filepath)
                
                # Hitung rata-rata kolom 'cer'
                rata_rata_cer = df['cer'].mean()
                
                # Format print agar terlihat rapi seperti tabel
                print(f"File: {filename: <40} | Avg CER: {rata_rata_cer:.4f}")
                
            except Exception as e:
                print(f"File: {filename: <40} | [ERROR] Gagal membaca: {e}")
        else:
            print(f"File: {filename: <40} | [INFO] File tidak ditemukan.")
            
    print("=" * 65)

if __name__ == '__main__':
    main()