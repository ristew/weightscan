import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 3), temporal_weight=1e6, lr=0.001, num_epochs=5, training_set=None):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 2048
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.temporal_weight = temporal_weight
        self.lr = lr
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_input = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.encoder_output = nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_input = nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim)
        self.decoder_hidden = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, input_dim)
        self.training_set = training_set

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
        return encoded, decoded

    def normalize(self, encoded):
        # Apply L2 normalization
        norm = (LA.norm(encoded, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_encoded = encoded / norm
        return normalized_encoded

    def temporal_penalty(self, encoded_prev, encoded_next):
        return F.mse_loss(encoded_prev, encoded_next) * self.temporal_weight

    def train_sample(self, sample):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        layer = 0
        sum_loss = 0
        for data in sample:
            optimizer.zero_grad()
            encoded, decoded = self(data.float())
            loss = self.criterion(decoded, data.float())
            temporal_loss = 0
            if layer != 0:
                temporal_loss = self.temporal_penalty(prev_encoded, encoded)
            total_loss = loss + temporal_loss
            print(f'layer {layer} loss {loss.item()} temporal {temporal_loss} total {total_loss.item()}')
            total_loss.backward(retain_graph=True)
            optimizer.step()
            prev_encoded = encoded.detach()
            layer += 1
            sum_loss += total_loss.item()
        print(f'sum loss: {sum_loss}')

    def train(self):
        prev_encoded = None
        for epoch in range(self.num_epochs):
            if epoch == self.num_epochs - self.num_epochs // 3:
                print('lr descend')
                self.lr = self.lr / 3
            for sample in self.training_set:
                self.train_sample(sample)