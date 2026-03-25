import torch.nn as nn
import torch.nn.functional as F

class ConvSubsampling(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.input_dim = input_dim
        self.d_model = d_model    
    
    def get_out_dim(self):
        dim = self.input_dim
        dim = (dim + 2 * 1 - 3) // 2 + 1
        dim = (dim + 2 * 1 - 3) // 2 + 1
        return self.d_model * dim
    
    def forward(self, x):
        # x: (batch, time, features) or (batch, channels, time, features)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, time, features)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, time, channels * freq)
        
        return x