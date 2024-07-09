import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from sklearn.neighbors import NearestNeighbors
from grokfast import gradfilter_ema

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 3), hidden_dim=2048, temporal_weight=1e6, lr=0.001, num_epochs=5, weight_decay=0):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = input_dim * 2;
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.temporal_weight = temporal_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1]),
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, input_dim),
        )

    def forward(self, x):
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder(x_pooled)
        encoded = a.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        encoded = self.normalize(encoded)
        a = encoded.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
        b = self.decoder(a)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded, x_pooled

    def normalize(self, encoded):
        norm = (LA.norm(encoded, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_encoded = encoded / norm
        return normalized_encoded

    def temporal_penalty(self, encoded_prev, encoded_next, state_prev, state):
        encoded_prev_norm = F.normalize(encoded_prev, p=2, dim=-1)
        encoded_next_norm = F.normalize(encoded_next, p=2, dim=-1)
        state_norm = F.normalize(state, p=2, dim=-1)
        prev_state_norm = F.normalize(state_prev, p=2, dim=-1)
        encoded_loss = torch.dist(encoded_prev_norm, encoded_next_norm, p=2)
        state_loss = torch.dist(prev_state_norm, state_norm, p=2)
        loss_diff = encoded_loss - state_loss
        return self.temporal_weight * loss_diff.abs(), state_loss

    def train_sample(self, sample):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        layer = 0
        sum_loss = 0
        prev_encoded = None
        prev_state = None
        total_loss = torch.tensor(0.0, requires_grad=True)
        sum_state = 0
        for data in sample:
            optimizer.zero_grad()
            encoded, decoded, pooled = self(data.float())
            loss = self.criterion(decoded, pooled)
            temporal_loss = 0
            if layer > 0 and layer < len(sample) - 3:
                temporal_loss, state_loss = self.temporal_penalty(prev_encoded, encoded, prev_state, data)
                sum_state += state_loss
            prev_state = data.detach()
            total_loss = total_loss + loss + temporal_loss
            prev_encoded = encoded.detach()
            layer += 1
        total_loss.backward(retain_graph=True)
        self.grads = gradfilter_ema(self, grads=self.grads)
        optimizer.step()
        return total_loss, sum_state

    def train_set(self, training_set):
        self.prev_encoded = None
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            if epoch == self.num_epochs - self.num_epochs // 3:
                print('lr descend')
                self.lr = self.lr / 3
            self.grads = None
            for i, sample in enumerate(training_set):
                sample_loss, sum_state = self.train_sample(sample)
                print(f'{epoch}:{i}\tloss {sample_loss.item():.3f}\tstate change {sum_state:.3f}')
