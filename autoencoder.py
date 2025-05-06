import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from metrics_reporter import MetricsReporter

class Autoencoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_components: int = 2**15,   # C
                 hidden_dim: int = 768,
                 top_k: int = 64,
                 lr: float = 1e-4,
                 num_epochs: int = 25,
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

    def forward(self, x):                     # x: (B,T,D)
        B,T,D = x.shape
        k = min(self.top_k * T, self.n_components)
        x = x.float()
        logits_tok = self.encoder(x)          # (B,T,C)
        logits_sum = logits_tok.sum(1) / math.sqrt(T)   # (B,C)
        _, idx = torch.topk(logits_sum, k, dim=-1)   # (B,k)
        batch_ix = torch.arange(B, device=x.device).unsqueeze(-1)
        comps_k   = self.components[idx]                 # (B,k,3)
        comps_k   = F.normalize(comps_k + torch.randn_like(comps_k)*self.noise_factor, dim=-1)
        dec_k     = self.decoder(comps_k)                # (B,k,D)
        idx_exp   = idx.unsqueeze(1).expand(-1,T,-1)     # (B,T,k)
        weights_t = torch.gather(logits_tok, 2, idx_exp) # (B,T,k)
        weights_t = weights_t.softmax(-1) * math.sqrt(k)
        x_recon   = torch.einsum('btk,bkd->btd', weights_t, dec_k)  # (B,T,D)

        return x_recon, comps_k                    # plus whatever else

    def _loss(self, batch):
        r, points = self.forward(batch)          # (B,k,3), (B,D)
        L_f = self.faithful_alpha * F.mse_loss(r, batch)
        loss = L_f
        self.reporter.update(loss=loss, f=L_f)
        return loss

    def train_set(self, training_set):
        if self.mark > 0:
            training_set = training_set[:self.mark]
        self.reporter = MetricsReporter()
        self.noise_factor = 0.5 
        self.noise_scaling = 0.001**(1/self.num_epochs)
        self.train()
        for epoch in range(self.num_epochs):
            for sample in training_set:
                for layer_t in sample:
                    self.opt.zero_grad()
                    loss = self._loss(layer_t)
                    loss.backward()
                    self.opt.step()
            self.reporter.epoch_end(epoch)
            self.noise_factor *= self.noise_scaling 
            print(f"new noise factor: {self.noise_factor}")
        self.eval()
