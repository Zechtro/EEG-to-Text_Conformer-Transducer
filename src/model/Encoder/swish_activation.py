import torch
import torch.nn as nn

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)