import torch

class BeamDecoderChar:
    def __init__(self, model, tokenizer, beam_size=3, max_sym_per_frame=3):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.blank_id = 0
        
        # Parameter krusial untuk RNN-T: Mencegah model terjebak dalam infinite loop
        # (terus-menerus memprediksi karakter tanpa pernah memprediksi blank di satu waktu)
        self.max_sym_per_frame = max_sym_per_frame

    @torch.no_grad()
    def decode(self, eeg_input):
        self.model.eval()
        device = eeg_input.device
        
        # 1. Encoder Pass
        encoder_out = self.model.encoder(eeg_input) # (1, T, enc_dim)
        T = encoder_out.size(1)

        # Inisialisasi list 'B': Beams yang sudah siap maju ke frame waktu (t+1)
        # Format beam: (list_token_ids, cumulative_score)
        B = [([self.blank_id], 0.0)]

        for t in range(T):
            f_t = encoder_out[:, t:t+1, :] # Ambil 1 frame waktu (1, 1, enc_dim)
            
            # 'A' adalah beams yang akan diekspansi pada frame waktu SAAT INI (t)
            A = B 
            B = [] # Kosongkan B untuk menampung beam yang selesai di waktu 't'

            # Loop ekspansi karakter dalam frame yang sama
            for _ in range(self.max_sym_per_frame):
                new_A = []
                
                for token_ids, score in A:
                    # 3. Decoder Pass
                    # (Menggunakan seluruh riwayat karakter yang diprediksi sejauh ini)
                    y = torch.tensor([token_ids], device=device)
                    g_u, _ = self.model.decoder(y)
                    
                    # Kita hanya butuh representasi dari token paling terakhir
                    g_u_last = g_u[:, -1:, :] 

                    # 4. Joiner Pass
                    logits = self.model.joiner(f_t, g_u_last)
                    log_probs = torch.log_softmax(logits, dim=-1).view(-1)

                    # 5. Top-K expansion
                    top_log_probs, top_ids = torch.topk(log_probs, self.beam_size)

                    for i in range(len(top_ids)):
                        v = top_ids[i].item()
                        p = top_log_probs[i].item()
                        
                        if v == self.blank_id:
                            # ATURAN RNN-T: Jika tebakan adalah BLANK, 
                            # beam ini selesai untuk waktu 't'. Pindahkan ke B untuk waktu 't+1'
                            B.append((token_ids, score + p))
                        else:
                            # ATURAN RNN-T: Jika tebakan adalah KARAKTER,
                            # tambahkan karakter ke list, tapi TETAPKAN beam ini di A 
                            # agar bisa diekspansi lagi di frame waktu 't' yang sama.
                            new_A.append((token_ids + [v], score + p))
                
                # Jika semua beam memprediksi blank (new_A kosong), berhenti ekspansi di frame ini
                if not new_A:
                    break
                
                # Pruning untuk membatasi ukuran memori di waktu 't' saat ini
                A = sorted(new_A, key=lambda x: x[1], reverse=True)[:self.beam_size]
            
            # Jika batas max_sym_per_frame tercapai, tapi masih ada beam di A yang belum memprediksi blank,
            # kita paksa pindahkan mereka ke B agar tidak hilang dan bisa lanjut ke waktu t+1
            B.extend(A)

            # Pruning beam B sebelum masuk ke siklus t+1 selanjutnya
            B = sorted(B, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # Ambil path terbaik dari iterasi frame terakhir
        best_ids, _ = B[0]
        
        # Hapus token blank HANYA. Jangan kurangi indeksnya (-1).
        final_ids = [i for i in best_ids if i != self.blank_id]
        
        return self.tokenizer.int_to_text(final_ids)