import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x:torch.Tensor) -> torch.Tensor:
          return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

class FullAttnRes(nn.Module):
    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.w = nn.Parameter(torch.zeros(num_layers, d_model))
        self.norm = RMSNorm(d_model)

    
          