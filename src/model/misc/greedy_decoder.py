import torch
import torch.nn.functional as F

# TODO: ganti Beam
class GreedyDecoder:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.blank_idx = 0

    def decode(self, audio_input):
        if audio_input.dim() == 2:
            audio_input = audio_input.unsqueeze(0)
        with torch.no_grad():
            encoder_out = self.model.encoder(audio_input)
        
        encoder_out = encoder_out.squeeze(0)
        T = encoder_out.size(0)
        t = 0
        predictions = []
        decoder_input = torch.tensor([[self.blank_idx]], device=self.device)
        hidden_state = None
        decoder_out, hidden_state = self.model.decoder(decoder_input, hidden_state)
        
        max_symbols = T * 2 
        symbols_added = 0
        while t < T and symbols_added < max_symbols:
            enc_t = encoder_out[t].unsqueeze(0).unsqueeze(0)
            logits = self.model.joiner(enc_t, decoder_out)
            best_token = torch.argmax(logits, dim=-1).item()
            if best_token == self.blank_idx:
                t += 1
            else:
                predictions.append(best_token)
                next_in = torch.tensor([[best_token]], device=self.device)
                decoder_out, hidden_state = self.model.decoder(next_in, hidden_state)
                symbols_added += 1
        return self.tokenizer.int_to_text(predictions)