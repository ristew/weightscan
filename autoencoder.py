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
                 n_components: int = 2048,   # C
                 hidden_dim: int = 1536,
                 n_layers: int = 32,
                 top_k: int = 8,
                 top_octants: int = 7,
                 lr: float = 3e-4,
                 num_epochs: int = 5,
                 beta_min: float = 1e8,
                 alpha_simp: float = 2,
                 r_min: float = 1e-4,
                 faithful_alpha: float = 4):
        super().__init__()
        self.input_dim   = input_dim
        self.n_components = n_components
        self.top_k       = top_k
        self.top_octants = top_octants
        self.num_epochs  = num_epochs
        self.beta_min    = beta_min   # weight on minimality loss
        self.alpha_simp  = alpha_simp # weight on simplicity loss
        self.faithful_alpha = faithful_alpha
        self.r_min = r_min

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
        mask.scatter_(-1, idx, True)
        return mask

    def _octant_topk_mask(self, logits: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            c_sign = (self.components > 0).to(torch.int) # (C, 3)
            oct_id = (c_sign[:, 0] << 2) | (c_sign[:, 1] << 1) | c_sign[:, 2] # (C,)

        probs = logits.softmax(dim=-1)
        mass8 = probs @ F.one_hot(oct_id.long(), 8).float()
        _, best = mass8.topk(self.top_octants, dim=-1)
        keep = (oct_id.unsqueeze(0) == best.unsqueeze(-1)).any(dim=1)
        return keep


    # ------------------------------------------------------------------
    # radial projection
    def _radial_clip(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B,C,3) batch of weighted component vectors.
        Any point whose length > r_min is projected onto the unit sphere
        (length = 1); the rest are zeroed.  All ops are differentiable.
        """
        norms = pts.norm(dim=-1, keepdim=True)                       # (B,C,1)
        keep  = norms > self.r_min
        # normalise only the ones we keep; avoid /0 with clamp
        proj  = pts / norms.clamp_min(1e-6)
        return torch.where(keep, proj, torch.zeros_like(pts))

    def forward(self, x: torch.Tensor, layer_idx: int):
        """Return ``encoded, decoded, pooled`` to match original API.
        * encoded  –  (B,C,3) tensor of all mechanism points (for inspection).
        * decoded  –  reconstruction from *sparse* (top‑k) mix.
        * pooled   –  original pooled activations.
        """
        x = x.float()
        pooled = self.pool(x.transpose(1,2)).squeeze(-1)        # (B,input_dim)
        # pooled = pooled + self.layer_embed(torch.tensor(layer_idx, device=x.device))
        logits = self.encoder(pooled)                           # (B,C)

        # oct_mask = self._octant_topk_mask(logits)
        # logits = logits.masked_fill(~oct_mask, float('-inf'))
        #
        weights = logits.softmax(dim=-1)                 # (B,C)
        raw_pts = weights.unsqueeze(-1) * self.components        # (B,C,3)
        proj_pts = self._radial_clip(raw_pts)                    # (B,C,3)
        points_sparse = proj_pts.sum(dim=1)                      # (B,3)
        decoded = self.decoder(points_sparse)                   # (B,input_dim)

        encoded = proj_pts                                       # (B,C,3)
        return encoded, decoded, pooled

    def _loss(self, batch, layer_idx):
        encoded, decoded_sparse, pooled = self.forward(batch, layer_idx)
        L_f = self.faithful_alpha * F.mse_loss(decoded_sparse, pooled)
        loss = L_f # + L_simp + L_min + L_origin
        self.reporter.update(loss=loss)
        return loss

    def train_set(self, training_set, layers):
        self.reporter = MetricsReporter()
        self.train()
        for epoch in range(self.num_epochs):
            for sample in training_set:
                for layer_idx in layers:
                    layer_t = sample[layer_idx]
                    self.opt.zero_grad()
                    loss = self._loss(layer_t, layer_idx)
                    loss.backward()
                    self.opt.step()
            self.reporter.epoch_end(epoch)
        self.eval()
        print(self.components)
