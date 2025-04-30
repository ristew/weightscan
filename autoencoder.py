import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from metrics_reporter import MetricsReporter

class Autoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_components: int = 8192,   # C
                 hidden_dim: int = 4096,
                 n_layers: int = 32,
                 top_k: int = 256,
                 lr: float = 3e-4,
                 num_epochs: int = 5,
                 origin_alpha: float = 10,
                 faithful_alpha: float = 1):
        super().__init__()
        self.input_dim   = input_dim
        self.n_components = n_components
        self.top_k       = top_k
        self.num_epochs  = num_epochs
        self.origin_alpha = origin_alpha
        self.faithful_alpha = faithful_alpha

        self.layer_embed = nn.Embedding(n_layers, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_components)  # logits per component
        )

        # learnable 3‑D mechanism dictionary (C × 3)
        self.components = nn.Parameter(torch.randn(n_components, 3))

        self.decoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        x = x.float()
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        logits = self.encoder(pooled)                           # (B,C)
        weights = logits.softmax(dim=-1)                 # (B,C)
        weights = weights * math.sqrt(self.n_components)
        points = weights.unsqueeze(-1) * self.components        # (B,C,3)
        _, idx = torch.topk(logits, self.top_k, dim=-1) # (B,top_k)
        batch_ix = torch.arange(x.size(0), device=x.device).unsqueeze(-1)
        points = points[batch_ix, idx]
        decoded = self.decoder(points).sum(dim=1)
        return points, decoded, pooled

    def _loss(self, batch):
        encoded, decoded, pooled = self.forward(batch)
        L_f = self.faithful_alpha * F.mse_loss(decoded, pooled)
        md = encoded.norm(dim=-1).mean(dim=-1).squeeze()
        mr = (1.0 - md)**2
        L_origin = self.origin_alpha * mr
        loss = L_f + L_origin
        self.reporter.update(loss=loss, f=L_f, o=L_origin)
        return loss

    def train_set(self, training_set):
        self.reporter = MetricsReporter()
        self.train()
        for epoch in range(self.num_epochs):
            for sample in training_set:
                for layer_idx in range(1, len(sample) - 1):
                    layer_t = sample[layer_idx]
                    self.opt.zero_grad()
                    loss = self._loss(layer_t)
                    loss.backward()
                    self.opt.step()
            self.reporter.epoch_end(epoch)
        self.eval()
