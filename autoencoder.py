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
    def __init__(self, input_dim, compressed_dim=(1024, 6), hidden_dim=1536, lr=0.001, num_epochs=5, weight_decay=0, diff_factor=0, ann_factor=0):
        super(Autoencoder, self).__init__()
        # Note compressed_dim[1] is now 6 (3 for position, 3 for direction)
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.diff_factor = diff_factor
        self.ann_factor = ann_factor
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # self.encoder = nn.Sequential(
        #     nn.Linear(self.input_dim, compressed_dim[0] * compressed_dim[1]),
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(compressed_dim[0] * compressed_dim[1], input_dim),
        # )
        # Encoder with nonlinear layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim[0] * compressed_dim[1]),
        )
        
        # Decoder with nonlinear layers
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def split_vector_field(self, encoded):
        # Split the encoded tensor into positions and directions
        positions = encoded[..., :3]
        directions = encoded[..., 3:]
        # Normalize the direction vectors
        directions = F.normalize(directions, dim=-1)
        return positions, directions

    def manifold_vector_field_loss(self, positions, directions, epsilon=1e-5, neighbor_k=8):
        pdist = torch.cdist(positions, positions)
        _, neighbor_idx = torch.topk(pdist, k=neighbor_k + 1, largest=False)
        neighbor_idx = neighbor_idx[:, 1:]
        
        local_points = positions[neighbor_idx]
        centers = local_points.mean(dim=1, keepdim=True)
        centered = local_points - centers
        
        # Compute curvature - encourage points to form more curved surfaces
        rel_vectors = local_points - positions.unsqueeze(1)  # [batch, k, 3]
        curvature = torch.abs(torch.sum(rel_vectors, dim=1))  # [batch, 3]
        curvature_loss = 1e-2 * torch.norm(curvature, dim=1).mean()
        
        # Original manifold calculations
        cov = torch.bmm(centered.transpose(1, 2), centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        normals = eigenvectors[:, :, 0]
        alignments = torch.abs(torch.sum(directions * normals, dim=1))
        
        local_dirs = directions[neighbor_idx]
        dir_diffs = torch.norm(local_dirs - directions.unsqueeze(1), dim=2)
        smooth_loss = dir_diffs.mean(dim=1)
        alignments_loss = 4 * alignments.mean()
        smooth_loss = 1e-3 * smooth_loss.mean()
        # print('alignments', alignments_loss, 'smooth', smooth_loss, 'curve', curvature_loss)
        
        total_loss = alignments_loss + smooth_loss + curvature_loss
        
        return total_loss

    def normalize(self, encoded):
        # Only normalize the position part
        positions, directions = self.split_vector_field(encoded)
        norm = (LA.norm(positions, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_positions = 64 * positions / norm
        # Recombine with directions
        return torch.cat([normalized_positions, directions], dim=-1)

    def train_sample(self, sample, batch_layers=True):
        layer = 0
        prev_encoded = None
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_vf_loss = torch.tensor(0.0, requires_grad=True)
        
        for data in sample:
            self.optimizer.zero_grad()
            data = data.float()
            encoded, decoded, pooled = self(data)
            positions, directions = self.split_vector_field(encoded)
            
            reconstruction_loss = self.criterion(decoded, pooled)
                
            vf_loss = 0.5 * self.manifold_vector_field_loss(positions.squeeze(), directions.squeeze())
            layer_loss = reconstruction_loss + vf_loss

            if layer == 0 or layer == len(sample) - 1:
                layer_loss = layer_loss * 3
            
            total_loss = total_loss + layer_loss
            total_vf_loss = total_vf_loss + vf_loss
            
            if not batch_layers:
                layer_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.grads = gradfilter_ema(self, self.grads)
                
            prev_encoded = encoded.detach()
            layer += 1
            
        if batch_layers:
            batch_loss = total_loss + total_vf_loss
            batch_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.grads = gradfilter_ema(self, self.grads)
            
        return total_loss, total_vf_loss


    def view_points(self, t):
        return t.view(-1, self.compressed_dim[0], self.compressed_dim[1])
    def points_t(self, points):
        return points.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
    def forward(self, x):
        # print('x', x.shape)
        x_pooled = self.global_pool(x.transpose(1, 2))
        x_pooled = x_pooled.squeeze(-1)
        a = self.encoder(x_pooled)
        encoded = self.normalize(self.view_points(a))
        randn = torch.randperm(encoded.size()[1])
        encoded = encoded[:, randn]
        a = self.points_t(encoded)
        slide = torch.mean(torch.abs(a)).item() / 32
        a = a + torch.randn_like(a) * slide / 128 + torch.full_like(a, torch.rand(1).item() * slide - slide / 2)
        b = self.decoder(a)
        decoded = b.view(-1, self.input_dim)
        return encoded, decoded, x_pooled

    def train_set(self, training_set):
        self.prev_encoded = None
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.grads = None
            for i, sample in enumerate(training_set):
                sample_loss, vf_loss = self.train_sample(sample)
                print(f'{epoch}:{i}\tloss {sample_loss.item():.3f}\tvf {vf_loss.item():.3f}')
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
