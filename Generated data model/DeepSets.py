import torch.nn as nn
import torch

class DeepSets(nn.Module):
    """DeepSets model for set input processing"""
    def __init__(self, input_dim, hidden_dim, output_dim, aggregator='sum', dropout=0.0):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.aggregator = aggregator

        self.phi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, mask=None, return_per_point=False):
        # x: (B, N, D)
        B, N, D = x.shape
        x_flat = x.view(B * N, D)
        h_flat = self.psi(x_flat)                   # (B*N, hidden_dim)
        h = h_flat.view(B, N, -1)    
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            h = h * mask_expanded  # zero out padding

        if self.aggregator == 'sum':
            pooled = h.sum(dim=1)
        elif self.aggregator == 'mean':
            if mask is not None:
                # mean over real points only
                counts = mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)  # (B, 1)
                pooled = h.sum(dim=1) / counts
            else:
                pooled = h.mean(dim=1)
        elif self.aggregator == 'max':
            if mask is not None:
                h_masked = h.clone()
                h_masked[~mask.unsqueeze(-1).expand_as(h)] = -1e9
                pooled = h_masked.max(dim=1)[0]
            else:
                pooled = h.max(dim=1)[0]
        else:
            raise ValueError("Unknown aggregator")
        out = self.phi(pooled)                      # (B, output_dim)
        if return_per_point:
            return out, h                           # (B, output_dim), (B, N, hidden_dim)
        return out

# small MLP head mapping pair of embeddings -> scalar cost
class Head(nn.Module):
    """Simple MLP head for pair of embeddings to scalar cost"""
    def __init__(self, emb_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, u, v):
        x = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)
        return self.net(x).squeeze(1)