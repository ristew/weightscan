import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from sklearn.neighbors import NearestNeighbors
from grokfast import gradfilter_ema

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(128, 3), hidden_dim=2048, lr=0.001, num_epochs=5, weight_decay=0, layer_weight=1):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.layer_weight = layer_weight

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim[0] * compressed_dim[1])
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.layer_norm = nn.LayerNorm(self.compressed_dim)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        # x shape: (batch_size, num_tokens, input_dim)
        batch_size, num_tokens, _ = x.size()

        # Encode each token
        encoded = []
        for i in range(num_tokens):
            token_encoded = self.encoder(x[:, i, :])
            token_encoded = token_encoded.view(-1, self.compressed_dim[0], self.compressed_dim[1])
            token_encoded = self.layer_norm(token_encoded)
            encoded.append(token_encoded)

        encoded = torch.stack(encoded, dim=1)  # (batch_size, num_tokens, 128, 3)

        # Decode entire layer
        decoded = []
        for i in range(num_tokens):
            token_decoded = self.decoder(encoded[:, i, :].view(batch_size, -1))
            decoded.append(token_decoded)

        decoded = torch.stack(decoded, dim=1)  # (batch_size, num_tokens, input_dim)

        return encoded, decoded

    def train_layer(self, layer_data, optimizer):
        optimizer.zero_grad()
        encoded, decoded = self(layer_data)
        loss = self.criterion(decoded, layer_data) * self.layer_weight
        loss.backward(retain_graph=True)
        optimizer.step()
        return encoded, loss
