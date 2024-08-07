import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.neighbors import NearestNeighbors
from grokfast import gradfilter_ema

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 3), hidden_dim=2048, lr=0.001, num_epochs=5, weight_decay=0):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = input_dim * 2;
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1]),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], compressed_dim[0] * compressed_dim[1]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def view_points(self, t):
        return t.view(-1, self.compressed_dim[0], self.compressed_dim[1])
    def points_t(self, points):
        return points.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
    def forward(self, x):
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder(x_pooled)
        a = self.view_points(a)
        a = torch.fft.fftn(a).real
        a = self.points_t(a)
        a = self.encoder2(a)
        a = self.view_points(a)
        a = self.normalize(a)
        encoded = a
        a = self.points_t(a)
        a = a + torch.randn_like(a) * 0.25
        b = self.decoder(a)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded, x_pooled

    def normalize(self, encoded):
        norm = (LA.norm(encoded, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_encoded = encoded / norm
        return normalized_encoded

    def train_sample(self, sample):
        layer = 0
        prev_encoded = None
        prev_state = None
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_diff_loss = torch.tensor(0.0, requires_grad=True)
        sum_delaunay = 0
        for data in sample:
            self.optimizer.zero_grad()
            encoded, decoded, pooled = self(data.float())
            reconstruction_loss = self.criterion(decoded, pooled)
            if prev_encoded is not None:
                diff_loss = 5e3 * self.criterion(encoded, prev_encoded)
                total_loss = total_loss + reconstruction_loss + diff_loss
                total_diff_loss = total_diff_loss + diff_loss
            else:
                total_loss = total_loss + reconstruction_loss
            prev_encoded = encoded.detach()
            layer += 1
        total_loss.backward(retain_graph=True)
        self.grads = gradfilter_ema(self, grads=self.grads)
        self.optimizer.step()
        return total_loss, total_diff_loss

    def train_set(self, training_set):
        self.prev_encoded = None
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            # if epoch == self.num_epochs - self.num_epochs // 3:
            #     print('lr descend')
            #     self.lr /= 3
            #     self.weight_decay /= 10
            self.grads = None
            for i, sample in enumerate(training_set):
                sample_loss, diff_loss = self.train_sample(sample)
                print(f'{epoch}:{i}\tloss {sample_loss.item():.3f}\tdiff {diff_loss.item():.3f}')
            self.scheduler.step()
