import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as plt

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
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(nn.Linear(d_model, d_model*2),
                                 nn.GELU(),
                                 nn.Linear(d_model*2, d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))
        

layer_fns = nn.ModuleList([DummyLayer(D_model) for _ in range(num_layers)])
embedding = torch.randn(B, T, D_model)

print(f"Model: {num_layers} layers, d_model={D_model}")
print(f"Block AttnRes: block_size={Block_size}, num_blocks={num_layers // Block_size}")
print(f"Input: batch={B}, seq_len={T}")

full_model_zero = FullAttnRes(num_layers, D_model)

with torch.no_grad():
    _, weight_matrix_zero = full_model_zero(embedding, layer_fns)

print("Attention weights for layer 5 (zero-init, should be uniform):")
print(weight_matrix_zero[4, :5].numpy().round(3))
print(f"Sum: {weight_matrix_zero[4, :5].sum().item():.3f}")

full_model = FullAttnRes(num_layers, D_model)
block_model = BlockAttnRes(num_layers, D_model, Block_size)

with torch.no_grad():
    nn.init.normal_(full_model.w, std=0.5)
    nn.init.normal_(block_model.w, std=0.5)


with torch.no_grad():
    _, full_weights = full_model(embedding, layer_fns)
    _, block_records = block_model(embedding, layer_fns)


print("Full AttnRes weight matrix shape:", full_weights.shape)
print(f"Block AttnRes: {len(block_records)} layers recorded")
print(f"\nExample -- Full AttnRes, layer 8 weights over sources 0..8:")
print(full_weights[7, :8].numpy().round(3))

def plot_full_attnres_heatmap(weight_matrix, title="Full AttnRes", ax=None):
    """Plot the Full AttnRes weight matrix as a heatmap."""
    L = weight_matrix.shape[0]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Mask upper triangle (layer l can only see sources 0..l-1)
    mask = np.zeros_like(weight_matrix.numpy())
    for i in range(L):
        mask[i, i + 1:] = 1
    data = weight_matrix.numpy().copy()
    data[mask.astype(bool)] = np.nan

    im = ax.imshow(data, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Source Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_yticks(range(L))
    ax.set_yticklabels(range(1, L + 1))
    ax.set_xticks(range(weight_matrix.shape[1]))
    return im


def plot_block_attnres_heatmap(block_records, num_layers, block_size,
                                title="Block AttnRes", ax=None):
    """Plot the Block AttnRes weight matrix as a heatmap."""
    num_blocks = num_layers // block_size
    max_sources = num_blocks + 2  # blocks + embedding + possible partial

    data = np.full((num_layers, max_sources), np.nan)
    for layer_idx, num_sources, avg_weights in block_records:
        for s in range(num_sources):
            data[layer_idx, s] = avg_weights[s].item()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(data, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel('Block Index', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(1, num_layers + 1))
    ax.set_xticks(range(max_sources))

    # Draw block boundaries
    for b in range(1, num_blocks):
        ax.axhline(y=b * block_size - 0.5, color='gray',
                   linestyle='--', linewidth=0.8, alpha=0.7)
    return im

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# (a) Zero-initialized (uniform weights)
im0 = plot_full_attnres_heatmap(weight_matrix_zero,
                                 title="Full AttnRes\n(zero-init: uniform weights)",
                                 ax=axes[0])

# (b) Simulated trained Full AttnRes
im1 = plot_full_attnres_heatmap(full_weights,
                                 title="Full AttnRes\n(simulated trained queries)",
                                 ax=axes[1])

# (c) Simulated trained Block AttnRes
im2 = plot_block_attnres_heatmap(block_records, num_layers, Block_size,
                                  title=f"Block AttnRes (S={Block_size})\n(simulated trained queries)",
                                  ax=axes[2])

fig.colorbar(im1, ax=axes, shrink=0.6, label='Attention Weight')
plt.suptitle('Depth-wise Attention Weight Distributions',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("=== Property 1: Weights sum to 1 ===")
for l in [0, 4, 8, 15]:
    num_sources = l + 1
    w_sum = full_weights[l, :num_sources].sum().item()
    print(f"  Layer {l+1:2d} ({num_sources:2d} sources): sum = {w_sum:.6f}")

print("\n=== Property 2: Zero-init -> uniform weights ===")
for l in [0, 4, 8, 15]:
    num_sources = l + 1
    expected = 1.0 / num_sources
    actual = weight_matrix_zero[l, :num_sources]
    print(f"  Layer {l+1:2d}: expected {expected:.4f} each, "
          f"got {actual.numpy().round(4)}")
    

total_layer_params = sum(p.numel() for p in layer_fns.parameters())
full_attnres_params = sum(p.numel() for p in full_model.parameters())
block_attnres_params = sum(p.numel() for p in block_model.parameters())

print(f"Layer parameters (all {num_layers} layers): {total_layer_params:,}")
print(f"Full AttnRes parameters:  {full_attnres_params:,} "
      f"({full_attnres_params / total_layer_params * 100:.2f}% of layer params)")
print(f"Block AttnRes parameters: {block_attnres_params:,} "
      f"({block_attnres_params / total_layer_params * 100:.2f}% of layer params)")
print(f"\nAttnRes adds only {num_layers} x {D_model} = {num_layers * D_model:,} "
      f"query parameters ({num_layers * D_model / total_layer_params * 100:.3f}% overhead)")