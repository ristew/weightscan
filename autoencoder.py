import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.neighbors import NearestNeighbors
from grokfast import gradfilter_ema

class FFT(nn.Module):
    def forward(self, x):
        return torch.fft.fftn(x).real

class Autoencoder(nn.Module):
    def __init__(self, input_dim, compressed_dim=(1024, 3), hidden_dim=2048, lr=0.001, num_epochs=5, weight_decay=0, diff_factor=0, ann_factor=0):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = 1536
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.diff_factor = diff_factor
        self.ann_factor = ann_factor
        self.encoder = nn.Sequential(
            # FFT(),
            nn.Linear(self.input_dim, compressed_dim[0] * compressed_dim[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.95, 0.998))
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def view_points(self, t):
        return t.view(-1, self.compressed_dim[0], self.compressed_dim[1])
    def points_t(self, points):
        return points.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
    def forward(self, x):
        a = self.encoder(x)
        encoded = self.normalize(self.view_points(a))
        a = self.points_t(encoded)
        slide = torch.mean(torch.abs(a)).item() / 100
        a = a + torch.randn_like(a) * slide * 0.1 + torch.full_like(a, torch.rand(1).item() * slide - slide / 2)
        b = self.decoder(a)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded

    def normalize(self, encoded):
        norm = (LA.norm(encoded, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_encoded = 64 * encoded / norm
        return normalized_encoded

    def average_nearest_neighbor_loss(self, encoded):
        pairwise_distances = torch.cdist(encoded, encoded, p=2)
        mask = torch.eye(pairwise_distances.size(0), device=pairwise_distances.device).bool()
        pairwise_distances = pairwise_distances.masked_fill(mask, float('inf'))
        nearest_neighbor_distances, _ = torch.min(pairwise_distances, dim=1)
        return nearest_neighbor_distances.mean()

    def train_sample(self, sample, batch_layers=False):
        layer = 0
        prev_encoded = None
        prev_state = None
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_diff_loss = torch.tensor(0.0, requires_grad=True)
        total_ann_loss = torch.tensor(0.0, requires_grad=True)
        sum_delaunay = 0
        for data in sample:
            self.optimizer.zero_grad()
            data = data.squeeze(0)[0].float() # only look at the first token
            encoded, decoded = self(data)
            reconstruction_loss = self.criterion(decoded, data)
            if layer == 0 or layer == len(sample) - 1:
                reconstruction_loss = reconstruction_loss * 3
            layer_loss = reconstruction_loss
            if self.ann_factor > 0:
                ann_loss = self.ann_factor * self.average_nearest_neighbor_loss(encoded.squeeze())
                layer_loss += ann_loss
                total_ann_loss = total_ann_loss + ann_loss
            if self.diff_factor > 0 and prev_encoded is not None:
                diff_loss = self.diff_factor * self.criterion(encoded, prev_encoded)
                layer_loss += diff_loss
                total_diff_loss = total_diff_loss + diff_loss
            total_loss = total_loss + layer_loss
            if not batch_layers:
                layer_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.grads = gradfilter_ema(self, self.grads)
            prev_encoded = encoded.detach()
            layer += 1
        if batch_layers:
            batch_loss = total_loss + total_ann_loss + total_diff_loss
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.grads = gradfilter_ema(self, self.grads)
        return total_loss, total_diff_loss, total_ann_loss

    def train_set(self, training_set):
        self.prev_encoded = None
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.grads = None
            for i, sample in enumerate(training_set):
                sample_loss, diff_loss, ann_loss = self.train_sample(sample)
                print(f'{epoch}:{i}\tloss {sample_loss.item():.3f}\tdiff {diff_loss.item():.3f}\tann {ann_loss.item():.3f}')
            self.scheduler.step()
        self.save_checkpoint()

    def save_checkpoint(self, checkpoint_dir='weights', filename='checkpoint.pth'):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

    def load_checkpoint(self, checkpoint_dir='weights', filename='checkpoint.pth'):
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
