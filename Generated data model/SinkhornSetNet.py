import torch
import torch.nn as nn
import torch.nn.functional as F

class PointMLPEncoder(nn.Module):
    """Encoder mapping individual points to embeddings via MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape
        h = self.psi(x.view(B*N, D))  # (B*N, output_dim)
        return h.view(B, N, -1)

class SinkhornSetNet(nn.Module):
    """Sinkhorn-based set matching network"""
    def __init__(self, input_dim, hidden_dim, emb_dim, cost_type='dot', sinkhorn_reg=0.1, sinkhorn_iters=30):
        super().__init__()
        self.encoder = PointMLPEncoder(input_dim, hidden_dim, emb_dim)
        self.cost_type = cost_type
        self.reg = sinkhorn_reg
        self.niter = sinkhorn_iters
    
    def sinkhorn_batch(self, C, mask=None, reg=0.1, niter=30):
        """
        Simple log-domain Sinkhorn for batch of cost matrices
        mask: (B, N) boolean indicating valid points
        """
        B, N, M = C.shape # N=M usually
        
        # Convert cost to similarity
        K = torch.exp(-C / reg)  # (B, N, M)
        
        if mask is not None:
            # Create mass vectors based on REAL set sizes
            # sum(mask) gives number of valid points per batch item
            lengths = mask.sum(dim=1).float().view(B, 1, 1) # (B, 1, 1)
            
            # Marginal u: 1/length for valid, 0 for padded
            u_target = (mask.float().view(B, N, 1) / lengths).to(C.device)
            v_target = (mask.float().view(B, M, 1) / lengths).to(C.device) # Assuming A and B same size

            # Zero out K connections to padding to prevent transport
            # (B, N, 1) * (B, 1, M) broadcasts to (B, N, M)
            mask_mat = mask.unsqueeze(2) * mask.unsqueeze(1)
            K = K * mask_mat.float()
        else:
            u_target = torch.ones(B, N, 1, device=C.device) / N
            v_target = torch.ones(B, M, 1, device=C.device) / M

        u = u_target.clone()
        v = v_target.clone()

        for _ in range(niter):
            # u = u_target / (K @ v)
            # Add eps to avoid div by zero, apply mask to keep padding 0
            KV = torch.bmm(K, v)
            u = u_target / (KV + 1e-10)
            
            KTU = torch.bmm(K.transpose(1, 2), u)
            v = v_target / (KTU + 1e-10)

        T = u * K * v.transpose(1, 2)  # transport matrix
        cost = (T * C).sum(dim=(1, 2))  # total cost per batch
        return T, cost

    def build_cost_matrix(self, A_emb, B_emb, cost_type='dot'):
        """
        A_emb: (B, N, H)
        B_emb: (B, M, H)
        Returns: cost matrix C (B, N, M)
        """
        if cost_type == 'dot':
            C = 1.0 - torch.bmm(A_emb, B_emb.transpose(1, 2))  # (B, N, M)
        elif cost_type == 'euclid':
            C = torch.cdist(A_emb, B_emb, p=2)                 # (B, N, M)
        else:
            raise ValueError("Unknown cost type")
        return C

    def forward(self, A_batch, B_batch, mask=None):
        # 1. Embed points
        A_emb = F.normalize(self.encoder(A_batch), dim=-1)
        B_emb = F.normalize(self.encoder(B_batch), dim=-1)

        # 2. Compute cost matrix
        C = self.build_cost_matrix(A_emb, B_emb, cost_type=self.cost_type)

        # 3. Sinkhorn
        _, cost = self.sinkhorn_batch(C, mask=mask, reg=self.reg, niter=self.niter)
        return cost