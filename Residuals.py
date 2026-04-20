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
    
    def forward(self, embedding: torch.Tensor, 
                layer_fns: list[nn.Module]) -> tuple[torch.Tensor, torch.Tensor]:
        
        L = len(layer_fns)
        assert L == self.num_layers
        B, T, D = embedding.shape

        layer_outputs = [embedding]
        weight_matrix = torch.zeros(L, L+1)

        for l in range(L):
            sources = torch.stack(layer_outputs) #[l+1, B, T, D]
            weights = self.compute_weights(l, sources) #[L+1, B, T]

            h_l = torch.einsum('n b t, n b t d -> b t d', weights, sources)
            avg_w = weights.mean(dim=(1, 2)).detach()
            weight_matrix[l, :l + 1] = avg_w

            v_l = layer_fns[l](h_l)
            layer_outputs.append(v_l)
        return h_l, weight_matrix
    
class BlockAttnRes(nn.Module):
    def __init__(self, num_layers: int, d_model: int, block_size: int):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.block_size = block_size
        self.w = nn.Parameter(torch.zeros(num_layers, d_model))
        self.norm = RMSNorm(d_model)

    def _attend(self, sources: list[torch.Tensor], query: list[torch.Tensor]):
        V = torch.stack(sources)
        K = self.norm()
        logits = torch.einsum('d, n b t d -> n b t', query, K)
        weights = F.softmax(logits, dim=0)
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)
        return h, weights
    
    def forward(self, embedding: torch.Tensor, layer_fns: list[nn.Module]) -> tuple[torch.Tensor, torch.Tensor]:
        L = len(layer_fns)
        B, T, D = embedding.shape

        blocks = [embedding]
        partial_block = None
        weight_records = []

        for l in range(L):
            if partial_block is not None:
                sources = blocks + partial_block
            else:
                sources = list(blocks)
        
            h_l, weights = self._attend(sources, self.w[l])

            avg_w = weights.mean(dim=(1,2)).detach()
            weight_records.append((l, len(sources), avg_w))

            v_l = layer_fns[l](h_l)

            if partial_block is None:
                partial_block = v_l
            else:
                partial_block = partial_block + v_l
            
            if (l+1)%self.block_size == 0:
                blocks.append(partial_block)
                partial_block = None
    
        return h_l, weight_records

num_layers = 16
D_model = 128
Block_size = 4
B, T = 4, 32

class DummyLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, d_model*2),
                                 nn.GELU,
                                 nn.Linear(d_model*2, d_model))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(self.norm(x))
        

layer_fns = nn.ModuleList([DummyLayer(D_model) for _ in range(num_layers)])
embedding = torch.randn(B, T, D_model)

print(f"Model: {num_layers} layers, d_model={D_model}")
print(f"Block AttnRes: block_size={Block_size}, num_blocks={num_layers // Block_size}")
print(f"Input: batch={B}, seq_len={T}")