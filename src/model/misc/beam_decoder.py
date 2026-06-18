import torch
import torch.nn.functional as F

class BeamDecoderIndoGPT:
    def __init__(self, model, tokenizer, beam_size=5, max_sym_per_frame=3, blank_id=0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.blank_id = blank_id
        self.max_sym_per_frame = max_sym_per_frame # WAJIB ADA

    @torch.no_grad()
    def decode(self, eeg_input):
        self.model.eval()
        device = eeg_input.device
        
        # 1. Encoder Pass
        encoder_out = self.model.encoder(eeg_input) # (1, T, enc_dim)
        T = encoder_out.size(1)

        # 2. Inisialisasi Ruang Tunggu (B)
        B = [([self.blank_id], 0.0)]

        for t in range(T):
            f_t = encoder_out[:, t:t+1, :] 
            
            # Pindahkan semesta dari ruang tunggu (B) ke arena evaluasi (A)
            A = B 
            B = []

            # 3. Inner Loop: Memungkinkan model memprediksi banyak sub-word di 1 frame
            for sym_count in range(self.max_sym_per_frame):
                new_A = []
                
                for token_ids, score in A:
                    # Decoder Pass IndoGPT
                    y = torch.tensor([token_ids], device=device)
                    g_u, _ = self.model.decoder(y)
                    g_u_last = g_u[:, -1:, :] 

                    # Joiner Pass
                    logits = self.model.joiner(f_t, g_u_last)
                    log_probs = F.log_softmax(logits, dim=-1).view(-1)

                    # Top-K expansion
                    top_log_probs, top_ids = torch.topk(log_probs, self.beam_size)

                    for i in range(len(top_ids)):
                        v = top_ids[i].item()
                        p = top_log_probs[i].item()
                        
                        if v == self.blank_id:
                            # Jika Blank: Pindah ke B (Lanjut ke t+1)
                            B.append((token_ids, score + p))
                        else:
                            # Jika Sub-Word: Tetap di A (Bisa keluar sub-word lagi di frame 't' ini)
                            new_A.append((token_ids + [v], score + p))
                
                # Jika tidak ada yang menebak sub-word, hentikan loop frame ini
                if not new_A:
                    break
                
                # Pruning untuk list A
                A = sorted(new_A, key=lambda x: x[1], reverse=True)[:self.beam_size]
            
            # Pindahkan sisa yang belum blank di A agar tidak hangus, masuk ke t+1
            B.extend(A)
            B = sorted(B, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # 4. Ambil tebakan terbaik di akhir waktu
        best_ids, _ = B[0]
        
        # 5. Hapus Blank (0) dan Kembalikan Index ke Asalnya (-1) untuk IndoGPT
        final_ids = [i - 1 for i in best_ids if i > 0]
        
        return self.tokenizer.decode(final_ids).strip()