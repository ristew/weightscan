import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 16)):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 512
        self.num_epochs = 8
        self.criterion = nn.MSELoss()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_input = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_output = nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_input = nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder_input(x_pooled)
        a = torch.relu(a)
        a = self.encoder_output(a)
        encoded = a.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        b = self.decoder_input(a)
        b = torch.relu(b)
        b = self.decoder_output(b)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded

    def train(self, training_set):
        print('training autoencoder')
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        for i in range(self.num_epochs):
            for data in training_set:
                optimizer.zero_grad()
                encoded, decoded = self(data.float())
                loss = self.criterion(decoded, data.float())
                loss.backward(retain_graph=True)
                optimizer.step()
        print('autoencoder trained')
