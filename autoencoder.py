import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from metrics_reporter import MetricsReporter

class Autoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_components: int = 65536,   # C
                 hidden_dim: int = 768,
                 top_k: int = 1024,
                 lr: float = 3e-5,
                 num_epochs: int = 10,
                 mark: int = -1,
                 faithful_alpha: float = 1,
                 ):
        super().__init__()
        self.input_dim   = input_dim
        self.n_components = n_components
        self.top_k       = top_k
        self.num_epochs  = num_epochs
        self.mark = mark
        self.faithful_alpha = faithful_alpha
        self.noise_factor = 0.0

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
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.opt = torch.optim.AdamW(self.parameters(), lr=lr)

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
        points = points + torch.randn_like(points) * self.noise_factor * 0.1
        points = F.normalize(points, dim=-1)
        decoded = self.decoder(points).sum(dim=1)
        return points, decoded, pooled

    def _loss(self, batch):
        pts, decoded, pooled = self.forward(batch)          # (B,k,3), (B,D)
        L_f = self.faithful_alpha * F.mse_loss(decoded, pooled)
        loss = L_f
        self.reporter.update(loss=loss, f=L_f)
        return loss

    def train_set(self, training_set):
        if self.mark > 0:
            training_set = training_set[:self.mark]
        self.reporter = MetricsReporter()
        self.noise_factor = 1.0
        self.train()
        for epoch in range(self.num_epochs):
            for sample in training_set:
                for layer_t in sample:
                    self.opt.zero_grad()
                    loss = self._loss(layer_t)
                    loss.backward()
                    self.opt.step()
            self.reporter.epoch_end(epoch)
            self.noise_factor *= 0.5
        self.eval()
