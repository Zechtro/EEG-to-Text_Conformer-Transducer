import torch

class BeamDecoder:
    def __init__(self, model, tokenizer, beam_size=5, blank_id=0):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.blank_id = blank_id

    @torch.no_grad()
    def decode(self, eeg_input):
        self.model.eval()
        device = eeg_input.device
        
        # 1. Encoder Pass
        encoder_out = self.model.encoder(eeg_input) # (1, T_sub, enc_dim)
        T = encoder_out.size(1)

        # 2. Initialize Beam: (token_ids, cumulative_score)
        beams = [([self.blank_id], 0.0)]

        for t in range(T):
            new_beams = []
            # f_t must be (1, 1, enc_dim) for the Joiner forward logic
            f_t = encoder_out[:, t:t+1, :] 

            for token_ids, score in beams:
                # 3. Decoder Pass
                y = torch.tensor([token_ids], device=device)
                g_u, _ = self.model.decoder(y)
                
                # We only need the prediction based on the very last token
                g_u_last = g_u[:, -1:, :] 

                # 4. Joiner Pass -> logits (1, 1, 1, vocab)
                logits = self.model.joiner(f_t, g_u_last)
                log_probs = torch.log_softmax(logits, dim=-1).view(-1)

                # 5. Top-K expansion
                top_log_probs, top_ids = torch.topk(log_probs, self.beam_size + 1)

                for i in range(len(top_ids)):
                    v = top_ids[i].item()
                    p = top_log_probs[i].item()
                    
                    if v == self.blank_id:
                        new_beams.append((token_ids, score + p))
                    else:
                        new_beams.append((token_ids + [v], score + p))

            # Prune
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.beam_size]

        # Final best path
        best_ids, _ = beams[0]
        # Remove blanks and shift back for IndoGPT
        final_ids = [i - 1 for i in best_ids if i > 0]
        return self.tokenizer.decode(final_ids)