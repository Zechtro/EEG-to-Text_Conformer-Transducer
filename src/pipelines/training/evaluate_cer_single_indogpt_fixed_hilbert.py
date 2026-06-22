import os
import pandas as pd

def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 65)
    print("REKAP EVALUASI RATA-RATA CER EKSMPERIMEN HILBERT")
    print("=" * 65)
    
    for i in range(1, 2):

        filename = f"SUB1_hilbert_test_predictions_10_{i}_IndoGPT.csv"

        filepath = os.path.join(script_dir, filename)
        
        if os.path.exists(filepath):
            try:

                df = pd.read_csv(filepath)
                
                rata_rata_cer = df['cer'].mean()
                
                print(f"File: {filename: <40} | Avg CER: {rata_rata_cer:.4f}")
                
            except Exception as e:
                print(f"File: {filename: <40} | [ERROR] Gagal membaca: {e}")
        else:
            print(f"File: {filename: <40} | [INFO] File tidak ditemukan.")
            
    print("=" * 65)

if __name__ == '__main__':
    main()