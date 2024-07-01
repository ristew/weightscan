import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from sklearn.neighbors import NearestNeighbors
from grokfast import gradfilter_ema

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 3), temporal_weight=1e6, distance_weight=1, lr=0.001, num_epochs=5, training_set=None, logprob_fn=None, weight_decay=0):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 4096
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.temporal_weight = temporal_weight
        self.distance_weight = distance_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_input = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.encoder_output = nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_input = nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim)
        self.decoder_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, input_dim)
        self.training_set = training_set
        self.logprob_fn = logprob_fn

    def forward(self, x):
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder_input(x_pooled)
        a = torch.relu(a)
        a = self.encoder_hidden(a)
        a = torch.relu(a)
        a = self.encoder_output(a)
        encoded = a.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        encoded = self.normalize(encoded)
        a = encoded.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
        b = self.decoder_input(a)
        b = torch.relu(b)
        a = self.decoder_hidden(b)
        b = torch.relu(b)
        b = self.decoder_output(b)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded, x_pooled

    def normalize(self, encoded):
        # Apply L2 normalization
        norm = (LA.norm(encoded, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_encoded = encoded / norm
        return normalized_encoded
    def temporal_penalty(self, encoded_prev, encoded_next, state):
        # Normalize vectors
        encoded_prev_norm = F.normalize(encoded_prev, p=2, dim=-1)
        encoded_next_norm = F.normalize(encoded_next, p=2, dim=-1)
        state_norm = F.normalize(state, p=2, dim=-1)
        prev_state_norm = F.normalize(self.prev_state, p=2, dim=-1)

        # Calculate losses using normalized vectors
        encoded_loss = F.mse_loss(encoded_prev_norm, encoded_next_norm)**2
        state_loss = F.mse_loss(prev_state_norm, state_norm)

        # Calculate the difference
        loss_diff = encoded_loss - state_loss

        return self.temporal_weight * loss_diff.abs()

    def train_sample(self, sample):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        layer = 0
        sum_loss = 0
        for data in sample:
            optimizer.zero_grad()
            encoded, decoded, pooled = self(data.float())
            loss = self.criterion(decoded, pooled)
            temporal_loss = 0
            if layer > 1 and layer < len(sample) - 3:
                temporal_loss = self.temporal_penalty(self.prev_encoded, encoded, data)
            self.prev_logprobs = self.logprob_fn(data)
            self.prev_state = data
            total_loss = loss + temporal_loss
            print(f'l{layer}\tloss {loss.item():.3g}\ttemporal {temporal_loss:.3g}\ttotal {total_loss.item():.3g}')
            total_loss.backward(retain_graph=True)
            self.grads = gradfilter_ema(self, grads=self.grads)
            optimizer.step()
            self.prev_encoded = encoded.detach()
            layer += 1
            sum_loss += total_loss.item()
        print(f'sum loss: {sum_loss}')

    def train_set(self):
        self.prev_encoded = None
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            if epoch == self.num_epochs - self.num_epochs // 3:
                print('lr descend')
                self.lr = self.lr / 3
            self.grads = None
            for sample in self.training_set:
                self.train_sample(sample)
