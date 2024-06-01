import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 16), sparsity_target=0.001, sparsity_weight=0.01, temporal_weight=1e6, lr=0.001, num_epochs=5):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 2048
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.temporal_weight = temporal_weight
        self.lr = lr
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.encoder_input = nn.Linear(input_dim, self.hidden_dim)
        self.encoder_output = nn.Linear(self.hidden_dim, compressed_dim[0] * compressed_dim[1])
        self.decoder_input = nn.Linear(compressed_dim[0] * compressed_dim[1], self.hidden_dim)
        self.decoder_output = nn.Linear(self.hidden_dim, input_dim)

    def forward(self, x):
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder_input(x_pooled)
        a = torch.relu(a)
        a = self.encoder_output(a)
        encoded = a.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        encoded = self.normalize(encoded)  # Normalize the encoded state
        b = self.decoder_input(a)
        b = torch.relu(b)
        b = self.decoder_output(b)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded

    def normalize(self, encoded):
        # Apply L2 normalization
        norm = torch.norm(encoded, p=2, dim=1, keepdim=True)
        normalized_encoded = encoded / norm
        return normalized_encoded

    def sparsity_penalty(self, encoded):
        mean_activation = torch.mean(encoded, dim=0)
        target = torch.full_like(mean_activation, self.sparsity_target)
        kl_div = F.kl_div(F.log_softmax(mean_activation, dim=0), target, reduction='sum')
        return kl_div

    def temporal_penalty(self, encoded_prev, encoded_next):
        return F.mse_loss(encoded_prev, encoded_next)

    def train(self, training_set):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        prev_encoded = None
        for epoch in range(self.num_epochs):
            layer = 0
            for data in training_set:
                optimizer.zero_grad()
                encoded, decoded = self(data.float())
                loss = self.criterion(decoded, data.float())
                sparsity_loss = self.sparsity_penalty(encoded)
                temporal_loss = 0
                if layer != 0:
                    temporal_loss = self.temporal_penalty(prev_encoded, encoded)
                total_loss = loss + min(self.sparsity_weight * sparsity_loss, loss) + min(self.temporal_weight * temporal_loss, loss)
                print(f'layer {layer} loss {loss.item()} sparsity {sparsity_loss.item() * self.sparsity_weight} temporal {temporal_loss * self.temporal_weight} total {total_loss.item()}')
                total_loss.backward(retain_graph=True)
                optimizer.step()
                prev_encoded = encoded.detach()
                layer += 1
            print(f'Epoch {epoch+1}, Loss: {total_loss.item()}')
