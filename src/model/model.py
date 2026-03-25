import torch.nn as nn
import sys
import os

current_file_path = os.path.abspath(__file__)
model_dir = os.path.dirname(current_file_path)

sys.path.append(os.path.join(model_dir, 'Encoder'))
sys.path.append(os.path.join(model_dir, 'Decoder'))
sys.path.append(os.path.join(model_dir, 'Joiner'))

from conformer import Conformer
from LSTM.decoder import LSTMDecoder
from IndoGPT.decoder import IndoGPTDecoder
from joiner import JointNetwork

class ConformerTransducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Conformer(
            input_dim=config['input_dim'], d_model=config['encoder_dim'],
            num_heads=4, num_layers=8, conv_kernel_size=31, dropout=0.1
        )
        self.decoder = LSTMDecoder(
            vocab_size=config['vocab_size'], embedding_dim=256,
            hidden_dim=config['decoder_dim'], num_layers=1, dropout=0.1
        )
        self.joiner = JointNetwork(
            vocab_size=config['vocab_size'], encoder_dim=config['encoder_dim'],
            decoder_dim=config['decoder_dim'], joint_dim=config['joint_dim']
        )

    def forward(self, audio_inputs, text_inputs):
        enc_out = self.encoder(audio_inputs)   
        dec_out, _ = self.decoder(text_inputs) 
        logits = self.joiner(enc_out, dec_out) 
        return logits

    def get_encoder_out_lengths(self, input_lengths):
        out = input_lengths
        out = (out + 2 * 1 - 3) // 2 + 1
        out = (out + 2 * 1 - 3) // 2 + 1
        return out
    
class ConformerIndoGPTTransducer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Conformer(input_dim=config['input_dim'], d_model=config['encoder_dim'])
        self.decoder = IndoGPTDecoder(joint_dim=config['decoder_dim'])
        self.joiner = JointNetwork(
            vocab_size=config['vocab_size'], 
            encoder_dim=config['encoder_dim'],
            decoder_dim=config['decoder_dim'], 
            joint_dim=config['joint_dim']
        )

    def forward(self, audio_inputs, text_inputs):
        enc_out = self.encoder(audio_inputs)   
        dec_out, _ = self.decoder(text_inputs) 
        return self.joiner(enc_out, dec_out)

    def get_encoder_out_lengths(self, input_lengths):
        out = (input_lengths + 2 * 1 - 3) // 2 + 1
        out = (out + 2 * 1 - 3) // 2 + 1
        return out