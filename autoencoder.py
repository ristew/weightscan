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
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, compressed_dim[0] * compressed_dim[1]),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(compressed_dim[0] * compressed_dim[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

    def temporal_consistency_loss(self, positions1, positions2, directions1, directions2, hidden1, hidden2, base_alpha=0.1, min_thresh=0.01):
        # Calculate hidden state differences and scale them meaningfully
        hidden_delta = torch.norm(hidden2 - hidden1, dim=-1)
        
        # Get percentile-based thresholds from the batch
        low_thresh = torch.quantile(hidden_delta, 0.25)
        high_thresh = torch.quantile(hidden_delta, 0.75)
        
        # Scale delta to be between min_thresh and 1.0 based on the distribution
        scaled_delta = min_thresh + (1.0 - min_thresh) * (hidden_delta - low_thresh) / (high_thresh - low_thresh + 1e-6)
        scaled_delta = torch.clamp(scaled_delta, min_thresh, 1.0)
        
        # Calculate actual position and direction changes
        position_delta = torch.norm(positions2 - positions1, dim=-1)
        direction_delta = torch.norm(directions2 - directions1, dim=-1)
        
        # Loss encourages movement proportional to hidden state changes
        position_loss = torch.mean(torch.abs(position_delta - scaled_delta))
        direction_loss = torch.mean(torch.abs(direction_delta - scaled_delta))
        
        return position_loss + direction_loss

    def flow_alignment_loss(self, positions1, positions2, directions1):
        movement = positions2 - positions1
        movement = F.normalize(movement, dim=-1)
        alignment = torch.sum(movement * directions1, dim=-1)
        return torch.mean(torch.abs(alignment))

    def spatial_frequency_bias(self, positions):
        # Take just x,y coordinates and reshape for 2D FFT
        positions_2d = positions[..., :2]  # Take first two coordinates
        fft = torch.fft.rfft2(positions_2d.float())
        
        # Create a simple frequency mask
        freq_mask = torch.ones_like(fft)
        if freq_mask.size(1) > 4:  # Only apply if we have enough frequencies
            mid_idx = freq_mask.size(1) // 2
            freq_mask[:, mid_idx-1:mid_idx+1] *= 2.0
        
        return torch.mean(torch.abs(fft * freq_mask))
    def split_vector_field(self, encoded):
        positions = encoded[..., :3]
        directions = encoded[..., 3:]
        directions = F.normalize(directions, dim=-1)
        return positions, directions

    def manifold_vector_field_loss(self, positions, directions, epsilon=1e-5, neighbor_k=8):
        pdist = torch.cdist(positions, positions)
        _, neighbor_idx = torch.topk(pdist, k=neighbor_k + 1, largest=False)
        neighbor_idx = neighbor_idx[:, 1:]
        
        local_points = positions[neighbor_idx]
        centers = local_points.mean(dim=1, keepdim=True)
        centered = local_points - centers
        
        rel_vectors = local_points - positions.unsqueeze(1)
        curvature = torch.abs(torch.sum(rel_vectors, dim=1))
        curvature_loss = 1e-2 * torch.norm(curvature, dim=1).mean()
        
        cov = torch.bmm(centered.transpose(1, 2), centered)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        normals = eigenvectors[:, :, 0]
        alignments = torch.abs(torch.sum(directions * normals, dim=1))
        
        local_dirs = directions[neighbor_idx]
        dir_diffs = torch.norm(local_dirs - directions.unsqueeze(1), dim=2)
        smooth_loss = dir_diffs.mean(dim=1)
        alignments_loss = 1 * alignments.mean()
        smooth_loss = 1e-3 * smooth_loss.mean()
        
        total_loss = alignments_loss + smooth_loss + curvature_loss
        
        return total_loss

    def normalize(self, encoded):
        positions, directions = self.split_vector_field(encoded)
        norm = (LA.norm(positions, ord=2, dim=1, keepdim=True) + 1) / 2
        normalized_positions = 64 * positions / norm
        return torch.cat([normalized_positions, directions], dim=-1)


    def train_sample(self, sample, batch_layers=True):
        layer = 0
        prev_positions = None
        prev_directions = None
        prev_pooled = None
        total_loss = torch.tensor(0.0, requires_grad=True)
        total_reconstruction_loss = 0
        total_vf_loss = 0
        total_temporal_loss = 0
        total_flow_loss = 0
        total_freq_loss = 0
        
        for data in sample:
            self.optimizer.zero_grad()
            data = data.float()
            encoded, decoded, pooled = self(data)
            positions, directions = self.split_vector_field(encoded)
            
            reconstruction_loss = self.criterion(decoded, pooled)
            total_reconstruction_loss = total_reconstruction_loss + reconstruction_loss.item()
            vf_loss = 0.1 * self.manifold_vector_field_loss(positions.squeeze(), directions.squeeze())
            total_vf_loss = total_vf_loss + vf_loss.item()
            
            if prev_positions is not None:
                temporal_loss = 0.04 * self.temporal_consistency_loss(
                    prev_positions, positions.squeeze(),
                    prev_directions, directions.squeeze(),
                    prev_pooled, pooled
                )
                flow_loss = 0.00 * self.flow_alignment_loss(
                    prev_positions, positions.squeeze(),
                    prev_directions
                )
                freq_loss = 0.000 * self.spatial_frequency_bias(positions.squeeze())
                
                total_temporal_loss = total_temporal_loss + temporal_loss.item()
                total_flow_loss = total_flow_loss + flow_loss.item()
                total_freq_loss = total_freq_loss + freq_loss.item()
                layer_loss = reconstruction_loss + vf_loss + temporal_loss + flow_loss + freq_loss
            else:
                layer_loss = reconstruction_loss + vf_loss
            
            prev_positions = positions.squeeze().detach()
            prev_directions = directions.squeeze().detach()
            prev_pooled = pooled.detach()
            
            total_loss = total_loss + layer_loss
            
            if not batch_layers:
                layer_loss.backward(retain_graph=True)
                self.optimizer.step()
                self.grads = gradfilter_ema(self, self.grads)
            
            layer += 1
            
        if batch_layers:
            total_loss.backward(retain_graph=True)
            self.optimizer.step()
            self.grads = gradfilter_ema(self, self.grads)
        
        print(f'total: {total_loss.item():.2f}  '
              f'loss: {total_reconstruction_loss:.2f}  '
              f'vf: {total_vf_loss:.2f}  '
              f'temporal: {total_temporal_loss:.2f}  '
              f'flow: {total_flow_loss:.2f}  '
              f'freq: {total_freq_loss:.2f}')

    def view_points(self, t):
        return t.view(-1, self.compressed_dim[0], self.compressed_dim[1])
        
    def points_t(self, points):
        return points.view(-1, self.compressed_dim[0] * self.compressed_dim[1])
        
    def forward(self, x):
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
                print(f'{epoch}:{i}', end=' ')
                self.train_sample(sample)
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
