import torch
import torch.nn as nn
from transformers import GPT2Model

# class IndoGPTDecoder(nn.Module):
#     def __init__(self, model_name_or_path="indobenchmark/indogpt", joint_dim=320):
#         super().__init__()
#         # Changed: use_safetensors=False because indogpt doesn't have a native safetensors version.
#         # Added: low_cpu_mem_usage=True for more efficient loading.
#         self.gpt = GPT2Model.from_pretrained(
#             model_name_or_path, 
#             use_safetensors=False,
#             low_cpu_mem_usage=True
#         )
        
#         # Skip bagian ini kalo modelnya mau ikut di train
#         for param in self.gpt.parameters():
#             param.requires_grad = False
            
#         self.output_proj = nn.Linear(self.gpt.config.n_embd, joint_dim)

#     def forward(self, y, hidden=None):
#         """
#         Args:
#             y: (Batch, U) sequence of label IDs (shifted by +1 for RNN-T blank).
#             hidden: Not used for GPT models, but kept for Transducer interface compatibility.
#         """
#         # Shift back: RNN-T label 1 -> GPT index 0. 
#         # Label 0 in RNN-T is Blank, which GPT shouldn't see as a token.
#         y_for_gpt = (y - 1).clamp(min=0).long() 
        
#         gpt_out = self.gpt(input_ids=y_for_gpt).last_hidden_state
        
#         # Project GPT's n_embd (768) to Joint dimension (320)
#         return self.output_proj(gpt_out), hidden


class IndoGPTDecoder(nn.Module):
    def __init__(self, model_name_or_path="indobenchmark/indogpt", joint_dim=320, context_size=None):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained(
            model_name_or_path, 
            use_safetensors=False,
            low_cpu_mem_usage=True
        )
        
        # Freezing GPT parameters
        for param in self.gpt.parameters():
            param.requires_grad = False
            
        self.output_proj = nn.Linear(self.gpt.config.n_embd, joint_dim)
        
        # N = How many previous tokens the model can "remember"
        self.context_size = context_size

    def forward(self, y, hidden=None):
        """
        y: (Batch, U) sequence of label IDs
        """
        y_for_gpt = (y - 1).clamp(min=0).long() 
        
        
        if self.context_size is not None:
            device = y.device
            batch_size, seq_len = y.size()

            # 1. Prepare IDs (Same as your original logic)
            
            # 2. CREATE SLIDING WINDOW MASK
            # Standard Causal Mask: (seq_len, seq_len) lower triangular
            full_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
            
            # Sliding Window Constraint: Only allow N tokens back
            # This creates a diagonal band of 1s
            # torch.triu with a negative diagonal clips the bottom part of the triangle
            # e.g., diagonal=-2 means we keep the current token + 2 tokens back
            window_mask = torch.triu(full_mask, diagonal=-self.context_size)
            
            # 3. Format Mask for Hugging Face
            # GPT2 expects attention_mask as (batch, seq_len) or (batch, 1, seq_len, seq_len)
            # We use the 4D version to pass our custom sliding window structure
            extended_mask = window_mask.unsqueeze(0).unsqueeze(0) # (1, 1, U, U)
            extended_mask = extended_mask.expand(batch_size, -1, -1, -1)
            
            # Transformers library uses 0 for mask out and 1 for keep
            # (Internally it converts 0 to -10000.0)
            
            # 4. GPT Forward Pass with Custom Mask
            gpt_out = self.gpt(
                input_ids=y_for_gpt,
                attention_mask=extended_mask
            ).last_hidden_state
        else:
            gpt_out = self.gpt(input_ids=y_for_gpt).last_hidden_state

        
        return self.output_proj(gpt_out), hidden