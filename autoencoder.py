import torch
import torch.nn as nn
import torch.nn.functional as F
from metrics_reporter import MetricsReporter

__all__ = ["Autoencoder"]

class Autoencoder(nn.Module):
    """Minimal APD‑style auto‑encoder that replaces the old dense latent with a *bank*
    of 3‑D mechanisms and a sparse gating network.
    The public interface is kept compatible with the original code used in
    ``train.py`` and ``visualize.py`` (forward returns ``encoded, decoded, pooled``).
    """

    def __init__(self,
                 input_dim: int,
                 n_components: int = 4096,   # C
                 hidden_dim: int = 1536,
                 n_layers: int = 32,
                 top_k: int = 100,
                 top_octants: int = 7,
                 lr: float = 3e-4,
                 num_epochs: int = 5,
                 r_min: float = 1e-5,
                 faithful_alpha: float = 1):
        super().__init__()
        self.input_dim   = input_dim
        self.n_components = n_components
        self.top_k       = top_k
        self.top_octants = top_octants
        self.num_epochs  = num_epochs
        self.faithful_alpha = faithful_alpha
        self.r_min = r_min

        # === modules ======================================================
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
        mask.scatter_(-1, idx, True)
        return mask

    # ------------------------------------------------------------------
    # radial projection
    def _radial_clip(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B,C,3) batch of weighted component vectors.
        Any point whose length > r_min is projected onto the unit sphere
        (length = 1); the rest are zeroed.  All ops are differentiable.
        """
        norms  = pts.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # (B,C,1)
        scale  = torch.where(norms > 1.0, 1.0 / norms, torch.ones_like(norms))
        return pts * scale
        # keep  = norms > self.r_min
        # # normalise only the ones we keep; avoid /0 with clamp
        # proj  = pts / norms.clamp_min(1e-6)
        # return torch.where(keep, proj, torch.zeros_like(pts))

    def forward(self, x: torch.Tensor, layer_idx: int):
        """Return ``encoded, decoded, pooled`` to match original API.
        * encoded  –  (B,C,3) tensor of all mechanism points (for inspection).
        * decoded  –  reconstruction from *sparse* (top‑k) mix.
        * pooled   –  original pooled activations.
        """
        x = x.float()
        pooled = x[:, -1, :] if x.dim() == 3 else x # (B,input_dim)
        logits = self.encoder(pooled)                           # (B,C)

        mask = self._batch_topk_mask(logits)
        logits = logits.masked_fill(~mask, float('-inf'))
        weights = logits.softmax(dim=-1)                 # (B,C)
        l2_norm = weights.norm(p=2, dim=-1, keepdim=True) # (B,1)
        weights = weights / (l2_norm + 1e-6)
        points = weights.unsqueeze(-1) * self.components        # (B,C,3)
        _, idx = torch.topk(logits, self.top_k, dim=-1) # (B,top_k)
        batch_ix = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        points = points[batch_ix, idx]
        decoded = self.decoder(points)                   # (B,input_dim)
        decoded = decoded.sum(dim=1)
        encoded = points                                      # (B,topk,3)
        return encoded, decoded, pooled

    def _loss(self, batch, layer_idx):
        encoded, decoded, pooled = self.forward(batch, layer_idx)
        L_f = self.faithful_alpha * F.mse_loss(decoded, pooled)
        loss = L_f # + L_simp + L_min + L_origin
        self.reporter.update(loss=loss)
        return loss

    def train_set(self, training_set, layer_idx):
        self.reporter = MetricsReporter()
        self.train()
        for epoch in range(self.num_epochs):
            for sample in training_set:
                layer_t = sample[layer_idx]
                self.opt.zero_grad()
                loss = self._loss(layer_t, layer_idx)
                loss.backward()
                self.opt.step()
            self.reporter.epoch_end(epoch)
        self.eval()
        print(self.components)
