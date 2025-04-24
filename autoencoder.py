import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Autoencoder"]

class Autoencoder(nn.Module):
    """Minimal APD‑style auto‑encoder that replaces the old dense latent with a *bank*
    of 3‑D mechanisms and a sparse gating network.
    The public interface is kept compatible with the original code used in
    ``train.py`` and ``visualize.py`` (forward returns ``encoded, decoded, pooled``).
    """

    def __init__(self,
                 input_dim: int,
                 n_components: int = 256,   # C
                 hidden_dim: int = 1536,
                 n_layers: int = 32,
                 top_k: int = 4,
                 lr: float = 3e-4,
                 num_epochs: int = 5,
                 beta_min: float = 1.0,
                 alpha_simp: float = 5e-3):
        super().__init__()
        self.input_dim   = input_dim
        self.n_components = n_components
        self.top_k       = top_k
        self.num_epochs  = num_epochs
        self.beta_min    = beta_min   # weight on minimality loss
        self.alpha_simp  = alpha_simp # weight on simplicity loss

        # === modules ======================================================
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.layer_embed = nn.Embedding(n_layers, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_components)  # logits per component
        )

        # learnable 3‑D mechanism dictionary (C × 3)
        self.components = nn.Parameter(torch.randn(n_components, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # helpers
    def _batch_topk_mask(self, scores: torch.Tensor):
        """Straight‑through binary mask selecting *top_k* components per batch row."""
        vals, idx = torch.topk(scores, self.top_k, dim=-1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        return mask

    # ------------------------------------------------------------------
    # forward
    def _dense_mix(self, logits):
        weights = logits.softmax(dim=-1)            # (B,C)
        return weights @ self.components            # (B,3)

    def _sparse_mix(self, logits, mask):
        logits_masked = logits.masked_fill(~mask, float('-inf'))
        weights = logits_masked.softmax(dim=-1)     # (B,C)
        return weights @ self.components            # (B,3)

    def forward(self, x: torch.Tensor, layer_idx: int):
        """Return ``encoded, decoded, pooled`` to match original API.
        * encoded  –  (B,C,3) tensor of all mechanism points (for inspection).
        * decoded  –  reconstruction from *sparse* (top‑k) mix.
        * pooled   –  original pooled activations.
        """
        x = x.float()
        # pooled = self.pool(x.transpose(1,2)).squeeze(-1)        # (B,input_dim)
        x = x + self.layer_embed(torch.tensor(layer_idx, device=x.device))
        logits = self.encoder(x)                           # (B,C)

        # -----   sparse mix   -------------------------------------------
        mask   = self._batch_topk_mask(logits)
        points_sparse = self._sparse_mix(logits, mask)          # (B,3)
        decoded = self.decoder(points_sparse)                   # (B,input_dim)

        # encoded: expose every component as a point for caller visualisation
        # Shape: (B,C,3)
        encoded = self.components.unsqueeze(0).expand(x.size(0), -1, -1)
        return encoded, decoded

    # ------------------------------------------------------------------
    # losses (APD‑inspired)
    def _loss(self, batch, layer_idx):
        encoded, decoded_sparse = self.forward(batch, layer_idx)

        # Dense reconstruction for faithfulness/minimality reference
        logits = self.encoder(batch)               # (B,C)
        points_dense = self._dense_mix(logits)      # (B,3)
        decoded_dense = self.decoder(points_dense)  # (B,input_dim)

        # 1) faithfulness (to pooled activations)
        L_f = F.mse_loss(decoded_dense, batch)
        # 2) minimality (sparse ≈ dense)
        L_min = F.mse_loss(decoded_sparse, decoded_dense)
        # 3) simplicity (Schatten‑½ on active components)
        mask = self._batch_topk_mask(logits).any(dim=0)         # (C,)
        active = self.components[mask]                          # (A,3)
        if active.numel() == 0:
            L_simp = torch.tensor(0., device=batch.device)
        else:
            L_simp = (active.norm(dim=1) ** 0.5).sum()

        loss = L_f + self.beta_min * L_min + self.alpha_simp * L_simp
        return loss

    # ------------------------------------------------------------------
    # public train API (minimal drop‑in for train.py)
    def train_set(self, training_set):
        self.train()
        et = 0
        for epoch in range(self.num_epochs):
            et += 1
            st = 0
            for sample in training_set:      # sample is a *list* of tensors
                st += 1
                sl = 0
                lt = 0
                for layer_t in sample:
                    self.opt.zero_grad()
                    loss = self._loss(layer_t, lt)
                    loss.backward()
                    self.opt.step()
                    sl += loss.item()
                    lt += 1
                print(f"loss@{et}:{st}={sl}")
        self.eval()
        print()
