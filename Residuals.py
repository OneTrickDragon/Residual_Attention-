import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def compute_weights(self, layer_idx: int, sources: torch.Tensor) -> torch.Tensor:
        K = self.norm(sources)
        q = self.norm(layer_idx)
        logits = torch.einsum('d, n b t d -> n b t', q, K)
        
        return F.softmax(logits, dim=0)