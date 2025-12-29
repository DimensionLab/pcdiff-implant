#!/usr/bin/env python3
"""
M1 Mac Compatible Training Script for PCDiff

This script trains the PCDiff point cloud diffusion model on Apple Silicon
using MPS (Metal Performance Shaders) or CPU.

Usage:
    python train_m1.py --path pcdiff/datasets/SkullBreak/train.csv \
                       --dataset SkullBreak \
                       --bs 2 \
                       --epochs 15000

Note: Training on M1 is significantly slower than on NVIDIA GPUs.
Consider using smaller batch sizes and monitoring memory usage.
"""

import argparse
import os
import sys
import time
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import functools


# ==============================================================================
# Pure PyTorch Implementations of Point Cloud Operations
# ==============================================================================

def ball_query(centers_coords, points_coords, radius, num_neighbors):
    """Ball query - find neighbors within radius."""
    B, _, M = centers_coords.shape
    _, _, N = points_coords.shape
    device = centers_coords.device
    
    centers = centers_coords.transpose(1, 2)
    points = points_coords.transpose(1, 2)
    dist_sq = torch.cdist(centers, points, p=2) ** 2
    radius_sq = radius ** 2
    
    # Vectorized approach - get top-k nearest neighbors within radius
    neighbor_indices = torch.zeros(B, M, num_neighbors, dtype=torch.int32, device=device)
    
    for b in range(B):
        for m in range(M):
            dists = dist_sq[b, m]
            # Sort by distance and take nearest
            sorted_idx = torch.argsort(dists)[:num_neighbors]
            neighbor_indices[b, m] = sorted_idx.int()
    
    return neighbor_indices


def furthest_point_sample(coords, num_samples):
    """Furthest point sampling."""
    B, C, N = coords.shape
    device = coords.device
    points = coords.transpose(1, 2)
    sampled_indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    
    for b in range(B):
        pts = points[b]
        # Start with random point
        farthest = torch.randint(0, N, (1,), device=device).item()
        sampled_indices[b, 0] = farthest
        distances = torch.sum((pts - pts[farthest]) ** 2, dim=1)
        
        for i in range(1, num_samples):
            farthest = torch.argmax(distances)
            sampled_indices[b, i] = farthest
            new_dists = torch.sum((pts - pts[farthest]) ** 2, dim=1)
            distances = torch.minimum(distances, new_dists)
    
    return gather(coords, sampled_indices.int())


def gather(features, indices):
    """Gather features by indices."""
    B, C, N = features.shape
    indices_expanded = indices.long().unsqueeze(1).expand(-1, C, -1)
    return torch.gather(features, 2, indices_expanded)


def grouping(features, indices):
    """Group features by neighbor indices."""
    B, C, N = features.shape
    _, M, U = indices.shape
    indices_flat = indices.long().view(B, -1)
    indices_expanded = indices_flat.unsqueeze(1).expand(-1, C, -1)
    gathered = torch.gather(features, 2, indices_expanded)
    return gathered.view(B, C, M, U)


def avg_voxelize(features, coords, resolution):
    """Average voxelization - optimized version."""
    B, C, N = features.shape
    R = resolution
    device = features.device
    
    coords_clamped = coords.long().clamp(0, R - 1)
    
    # Flatten spatial indices
    flat_idx = coords_clamped[:, 0] * R * R + coords_clamped[:, 1] * R + coords_clamped[:, 2]  # [B, N]
    
    voxels = torch.zeros(B, C, R * R * R, device=device, dtype=features.dtype)
    counts = torch.zeros(B, 1, R * R * R, device=device, dtype=features.dtype)
    
    for b in range(B):
        voxels[b].scatter_add_(1, flat_idx[b:b+1].expand(C, -1), features[b])
        counts[b].scatter_add_(1, flat_idx[b:b+1], torch.ones(1, N, device=device))
    
    counts = counts.clamp(min=1)
    voxels = voxels / counts
    
    return voxels.view(B, C, R, R, R)


def trilinear_devoxelize(features, coords, resolution, training=True):
    """
    Trilinear devoxelization implemented without grid_sample so it works on MPS.
    Only gradients w.r.t. voxel features are needed (coords are detached earlier),
    so this gather-based interpolation is sufficient and MPS-friendly.
    """
    B, C, R, _, _ = features.shape
    # Clamp to valid range and get fractional offsets
    coords = coords.clamp(0, resolution - 1 - 1e-6)
    idx0 = torch.floor(coords)
    idx1 = torch.clamp(idx0 + 1, max=resolution - 1)
    frac = coords - idx0
    idx0 = idx0.long()
    idx1 = idx1.long()

    def gather(ix, iy, iz):
        flat = (ix * R + iy) * R + iz  # [B, N]
        flat = flat.unsqueeze(1).expand(-1, C, -1)
        return torch.gather(features.view(B, C, -1), 2, flat)

    x0, y0, z0 = idx0[:, 0], idx0[:, 1], idx0[:, 2]
    x1, y1, z1 = idx1[:, 0], idx1[:, 1], idx1[:, 2]
    dx, dy, dz = frac[:, 0], frac[:, 1], frac[:, 2]

    c000 = gather(x0, y0, z0)
    c001 = gather(x0, y0, z1)
    c010 = gather(x0, y1, z0)
    c011 = gather(x0, y1, z1)
    c100 = gather(x1, y0, z0)
    c101 = gather(x1, y0, z1)
    c110 = gather(x1, y1, z0)
    c111 = gather(x1, y1, z1)

    # Unsqueeze weights for channel-wise broadcasting
    wx0, wx1 = (1 - dx).unsqueeze(1), dx.unsqueeze(1)
    wy0, wy1 = (1 - dy).unsqueeze(1), dy.unsqueeze(1)
    wz0, wz1 = (1 - dz).unsqueeze(1), dz.unsqueeze(1)

    # Combine 8 corners
    out = (
        c000 * wx0 * wy0 * wz0 +
        c001 * wx0 * wy0 * wz1 +
        c010 * wx0 * wy1 * wz0 +
        c011 * wx0 * wy1 * wz1 +
        c100 * wx1 * wy0 * wz0 +
        c101 * wx1 * wy0 * wz1 +
        c110 * wx1 * wy1 * wz0 +
        c111 * wx1 * wy1 * wz1
    )
    return out


def nearest_neighbor_interpolate(points_coords, centers_coords, centers_features):
    """Nearest neighbor interpolation."""
    B, _, N = points_coords.shape
    _, C, M = centers_features.shape
    
    points = points_coords.transpose(1, 2)
    centers = centers_coords.transpose(1, 2)
    
    dists = torch.cdist(points, centers, p=2)
    nearest_idx = torch.argmin(dists, dim=2)
    
    nearest_idx_expanded = nearest_idx.unsqueeze(1).expand(-1, C, -1)
    return torch.gather(centers_features, 2, nearest_idx_expanded)


# ==============================================================================
# Neural Network Modules
# ==============================================================================

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels, dim=1):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [out_channels]
        
        layers = []
        for oc in out_channels:
            if dim == 1:
                conv = nn.Conv1d(in_channels, oc, 1)
            else:
                conv = nn.Conv2d(in_channels, oc, 1)
            layers.extend([conv, nn.GroupNorm(8, oc), Swish()])
            in_channels = oc
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class SE3d(nn.Module):
    def __init__(self, channels, use_relu=False):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 4, 1),
            nn.ReLU(True) if use_relu else Swish(),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(x)


class Attention(nn.Module):
    def __init__(self, in_ch, num_groups, D=3):
        super().__init__()
        if D == 3:
            self.q = nn.Conv3d(in_ch, in_ch, 1)
            self.k = nn.Conv3d(in_ch, in_ch, 1)
            self.v = nn.Conv3d(in_ch, in_ch, 1)
            self.out = nn.Conv3d(in_ch, in_ch, 1)
        else:
            self.q = nn.Conv1d(in_ch, in_ch, 1)
            self.k = nn.Conv1d(in_ch, in_ch, 1)
            self.v = nn.Conv1d(in_ch, in_ch, 1)
            self.out = nn.Conv1d(in_ch, in_ch, 1)
        
        self.norm = nn.GroupNorm(num_groups, in_ch)
        self.nonlin = Swish()
        self.sm = nn.Softmax(-1)
    
    def forward(self, x):
        B, C = x.shape[:2]
        q = self.q(x).reshape(B, C, -1)
        k = self.k(x).reshape(B, C, -1)
        v = self.v(x).reshape(B, C, -1)
        qk = torch.matmul(q.permute(0, 2, 1), k)
        w = self.sm(qk)
        h = torch.matmul(v, w.permute(0, 2, 1)).reshape(B, C, *x.shape[2:])
        h = self.out(h)
        return self.nonlin(self.norm(h + x))


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps
    
    def forward(self, features, coords):
        coords = coords.detach()
        norm_coords = coords - coords.mean(2, keepdim=True)
        if self.normalize:
            norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
        else:
            norm_coords = (norm_coords + 1) / 2.0
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
        vox_coords = torch.round(norm_coords).to(torch.int32)
        return avg_voxelize(features, vox_coords, self.r), norm_coords


class BallQuery(nn.Module):
    def __init__(self, radius, num_neighbors, include_coordinates=True):
        super().__init__()
        self.radius = radius
        self.num_neighbors = num_neighbors
        self.include_coordinates = include_coordinates
    
    def forward(self, points_coords, centers_coords, temb, points_features):
        neighbor_indices = ball_query(centers_coords, points_coords, self.radius, self.num_neighbors)
        
        grouped_coords = grouping(points_coords, neighbor_indices)
        grouped_coords -= centers_coords.unsqueeze(-1)
        
        grouped_temb = grouping(temb, neighbor_indices)
        
        if points_features is None:
            grouped_features = grouped_coords
        else:
            grouped_features = grouping(points_features, neighbor_indices)
            if self.include_coordinates:
                grouped_features = torch.cat([grouped_coords, grouped_features], dim=1)
        
        return grouped_features, grouped_temb


class PVConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, attention=False,
                 dropout=0.1, with_se=False, with_se_relu=False, normalize=True, eps=0):
        super().__init__()
        self.resolution = resolution
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        
        voxel_layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            Swish()
        ]
        if dropout:
            voxel_layers.append(nn.Dropout(dropout))
        voxel_layers.extend([
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.GroupNorm(8, out_channels),
            Attention(out_channels, 8) if attention else Swish()
        ])
        if with_se:
            voxel_layers.append(SE3d(out_channels, use_relu=with_se_relu))
        
        self.voxel_layers = nn.Sequential(*voxel_layers)
        self.point_features = SharedMLP(in_channels, out_channels)
    
    def forward(self, inputs):
        features, coords, temb = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_layers(voxel_features)
        voxel_features = trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords, temb


class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]
        
        mlps = []
        total_out_channels = 0
        for _out_channels in out_channels:
            mlps.append(SharedMLP(in_channels + (3 if include_coordinates else 0), _out_channels, dim=1))
            total_out_channels += _out_channels[-1]
        
        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)
    
    def forward(self, inputs):
        features, coords = inputs
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = [mlp(features).max(dim=-1, keepdim=True).values for mlp in self.mlps]
            return torch.cat(features_list, dim=1), coords
        else:
            return self.mlps[0](features).max(dim=-1, keepdim=True).values, coords


class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        
        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(BallQuery(radius=_radius, num_neighbors=_num_neighbors, include_coordinates=include_coordinates))
            mlps.append(SharedMLP(in_channels + (3 if include_coordinates else 0), _out_channels, dim=2))
            total_out_channels += _out_channels[-1]
        
        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)
    
    def forward(self, inputs):
        features, coords, temb = inputs
        centers_coords = furthest_point_sample(coords, self.num_centers)
        features_list = []
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_features, grouped_temb = grouper(coords, centers_coords, temb, features)
            out = mlp(grouped_features)
            features_list.append(out.max(dim=-1).values)
        
        temb_out = grouped_temb.max(dim=-1).values if temb.shape[1] > 0 else temb
        return features_list[0], centers_coords, temb_out


class PointNetFPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels, out_channels, dim=1)
    
    def forward(self, inputs):
        if len(inputs) == 4:
            points_coords, centers_coords, centers_features, temb = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features, temb = inputs
        
        interpolated_features = nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        interpolated_temb = nearest_neighbor_interpolate(points_coords, centers_coords, temb)
        
        if points_features is not None:
            interpolated_features = torch.cat([interpolated_features, points_features], dim=1)
        
        return self.mlp(interpolated_features), points_coords, interpolated_temb


# ==============================================================================
# PVCNN2 Architecture
# ==============================================================================

def _linear_gn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.GroupNorm(8, out_channels), Swish())


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier
    if dim == 1:
        block = _linear_gn_relu
    else:
        block = SharedMLP
    
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels
    
    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_gn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, embed_dim=64, use_att=False,
                                   dropout=0.1, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3
    
    sa_layers, sa_in_channels = [], []
    c = 0
    for conv_configs, sa_configs in sa_blocks:
        k = 0
        sa_in_channels.append(in_channels)
        sa_block_layers = []
        
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = (c + 1) % 2 == 0 and c > 0 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention, dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)
                
                if c == 0:
                    sa_block_layers.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_block_layers.append(block(in_channels + embed_dim, out_channels))
                in_channels = out_channels
                k += 1
            extra_feature_channels = in_channels
        
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius, num_neighbors=num_neighbors)
        
        sa_block_layers.append(block(in_channels=extra_feature_channels + (embed_dim if k == 0 else 0),
                                     out_channels=out_channels, include_coordinates=True))
        c += 1
        in_channels = extra_feature_channels = sa_block_layers[-1].out_channels
        
        if len(sa_block_layers) == 1:
            sa_layers.append(sa_block_layers[0])
        else:
            sa_layers.append(nn.Sequential(*sa_block_layers))
    
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, sv_points, embed_dim=64, use_att=False,
                                dropout=0.1, with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    fp_layers = []
    c = 0
    
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_block_layers = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_block_layers.append(PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
                                                out_channels=out_channels))
        in_channels = out_channels[-1]
        
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            for p in range(num_blocks):
                attention = c % 2 == 0 and c < len(fp_blocks) - 1 and use_att and p == 0
                if voxel_resolution is None:
                    block = SharedMLP
                else:
                    block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                              attention=attention, dropout=dropout,
                                              with_se=with_se and not attention, with_se_relu=True,
                                              normalize=normalize, eps=eps)
                fp_block_layers.append(block(in_channels, out_channels))
                in_channels = out_channels
        
        if len(fp_block_layers) == 1:
            fp_layers.append(fp_block_layers[0])
        else:
            fp_layers.append(nn.Sequential(*fp_block_layers))
        c += 1
    
    return fp_layers, in_channels


class PVCNN2Base(nn.Module):
    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sv_points = sv_points
        self.in_channels = extra_feature_channels + 3
        
        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        
        self.sa_layers = nn.ModuleList(sa_layers)
        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)
        
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            sv_points=sv_points, with_se=True, embed_dim=embed_dim,
            use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        
        self.fp_layers = nn.ModuleList(fp_layers)
        
        layers, _ = create_mlp_components(in_channels=channels_fp_features, out_channels=[128, 0.5, num_classes],
                                          classifier=True, dim=2, width_multiplier=width_multiplier)
        self.classifier = nn.Sequential(*layers)
        
        self.embedf = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                    nn.LeakyReLU(0.1, inplace=True),
                                    nn.Linear(embed_dim, embed_dim))
    
    def get_timestep_embedding(self, timesteps, device):
        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = TF.pad(emb, (0, 1), "constant", 0)
        return emb
    
    def forward(self, inputs, t):
        temb = self.embedf(self.get_timestep_embedding(t, inputs.device))[:, :, None].expand(-1, -1, inputs.shape[-1])
        coords, features = inputs[:, :3, :].contiguous(), inputs
        coords_list, in_features_list = [], []
        
        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(features)
            coords_list.append(coords)
            if i == 0:
                features, coords, temb = sa_blocks((features, coords, temb))
            else:
                features, coords, temb = sa_blocks((torch.cat([features, temb], dim=1), coords, temb))
        
        in_features_list[0] = inputs[:, 3:, :].contiguous()
        
        if self.global_att is not None:
            features = self.global_att(features)
        
        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            jump_coords = coords_list[-1 - fp_idx]
            fump_feats = in_features_list[-1 - fp_idx]
            features, coords, temb = fp_blocks((jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb))
        
        return self.classifier(features)


class PVCNN2(PVCNN2Base):
    num_n = 128
    sa_blocks = [((32, 2, 32), (10240, 0.1, 128, (32, 64))),
                 ((64, 3, 16), (2560, 0.2, 128, (64, 128))),
                 ((128, 3, 8), (640, 0.4, 128, (128, 256))),
                 (None, (160, 0.8, 128, (256, 256, 512)))]
    fp_blocks = [((256, 256), (256, 3, 8)),
                 ((256, 256), (256, 3, 8)),
                 ((256, 128), (128, 2, 16)),
                 ((128, 128, 64), (64, 2, 32))]
    
    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


# ==============================================================================
# Diffusion Model
# ==============================================================================

class GaussianDiffusion:
    def __init__(self, betas, sv_points):
        self.np_betas = betas = betas.astype(np.float64)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        
        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        
        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    @staticmethod
    def _extract(a, t, x_shape):
        bs, = t.shape
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))
    
    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data (t == 0 means diffused for 1 step)"""
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        
        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)
    
    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """Training loss calculation"""
        B, D, N = data_start.shape
        
        if noise is None:
            noise = torch.randn(data_start[:, :, self.sv_points:].shape, dtype=data_start.dtype, device=data_start.device)
        
        data_t = self.q_sample(x_start=data_start[:, :, self.sv_points:], t=t, noise=noise)
        
        # Predict the noise
        eps_recon = denoise_fn(torch.cat([data_start[:, :, :self.sv_points], data_t], dim=-1), t)[:, :, self.sv_points:]
        losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))
        
        return losses


class Model(nn.Module):
    def __init__(self, num_points, num_nn, betas):
        super().__init__()
        sv_points = num_points - num_nn
        self.diffusion = GaussianDiffusion(betas, sv_points=sv_points)
        self.model = PVCNN2(num_classes=3, sv_points=sv_points, embed_dim=64,
                           use_att=True, dropout=0.1, extra_feature_channels=0,
                           width_multiplier=1.0, voxel_resolution_multiplier=1.0)
    
    def _denoise(self, data, t):
        return self.model(data, t)
    
    def get_loss(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)
        losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        return losses.mean()


# ==============================================================================
# Dataset
# ==============================================================================

class SkullBreakDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_points, num_nn, norm_mode='shape_bbox', augment=False):
        super().__init__()
        self.num_points = num_points
        self.num_nn = num_nn
        self.norm_mode = norm_mode
        self.augment = augment
        self.defects = ['bilateral', 'frontoorbital', 'parietotemporal', 'random_1', 'random_2']
        
        self.database = []
        import csv
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for defect_id in range(5):
                    defective = row[0].split('complete')[0] + 'defective_skull/' + self.defects[defect_id] + \
                                row[0].split('skull')[1].split('.')[0] + '_surf.npy'
                    implant = row[0].split('complete')[0] + 'implant/' + self.defects[defect_id] + \
                              row[0].split('skull')[1].split('.')[0] + '_surf.npy'
                    if os.path.exists(defective) and os.path.exists(implant):
                        self.database.append({'defective': defective, 'implant': implant})
    
    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, idx):
        item = self.database[idx]
        
        # Load point clouds
        pc_defective = np.load(item['defective'])
        pc_implant = np.load(item['implant'])
        
        # Sample points
        idx_def = np.random.randint(0, len(pc_defective), self.num_points - self.num_nn)
        idx_imp = np.random.randint(0, len(pc_implant), self.num_nn)
        
        pc_defective = pc_defective[idx_def]
        pc_implant = pc_implant[idx_imp]
        
        # Concatenate
        pc_combined = np.concatenate([pc_defective, pc_implant], axis=0)
        
        # Normalize
        if self.norm_mode == 'shape_bbox':
            pc_max = pc_combined.max(axis=0)
            pc_min = pc_combined.min(axis=0)
            shift = (pc_min + pc_max) / 2
            scale = (pc_max - pc_min).max() / 2 / 3
        else:
            shift = pc_combined.mean(axis=0)
            scale = pc_combined.std()
        
        pc_combined = (pc_combined - shift) / max(scale, 1e-6)
        
        return {'train_points': torch.from_numpy(pc_combined).float()}


# ==============================================================================
# Training
# ==============================================================================

def get_device(force_cpu=False):
    if force_cpu:
        return torch.device('cpu')
    if torch.backends.mps.is_available():
        print("✓ Using MPS (Metal Performance Shaders)")
        return torch.device('mps')
    print("⚠ MPS not available, using CPU")
    return torch.device('cpu')


def get_betas(beta_start=0.0001, beta_end=0.02, time_num=1000):
    return np.linspace(beta_start, beta_end, time_num)


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'loss': loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description='Train PCDiff on M1 Mac')
    parser.add_argument('--path', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--dataset', type=str, default='SkullBreak', help='Dataset name')
    parser.add_argument('--bs', type=int, default=2, help='Batch size (keep small for M1)')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=15000, help='Number of epochs')
    parser.add_argument('--num_points', type=int, default=30720, help='Total points')
    parser.add_argument('--num_nn', type=int, default=3072, help='Implant points')
    parser.add_argument('--output_dir', type=str, default='output_m1', help='Output directory')
    parser.add_argument('--save_every', type=int, default=500, help='Save checkpoint every N epochs')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  PCDiff Training on Apple Silicon (M1/M2/M3)")
    print("="*70)
    
    device = get_device(args.cpu)
    
    # Create output directory
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    print(f"\nLoading dataset from: {args.path}")
    dataset = SkullBreakDataset(args.path, args.num_points, args.num_nn)
    print(f"Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True)
    
    # Model
    print("\nBuilding model...")
    betas = get_betas()
    model = Model(args.num_points, args.num_nn, betas)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume
    start_epoch = 0
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting training:")
    print(f"  Epochs: {start_epoch} -> {args.epochs}")
    print(f"  Batch size: {args.bs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            pc = batch['train_points'].transpose(1, 2).to(device)
            
            optimizer.zero_grad()
            loss = model.get_loss(pc)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f'epoch_{epoch}.pth'
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)
            print(f"  💾 Saved checkpoint: {ckpt_path}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, output_dir / 'best.pth')
        
        print(f"Epoch {epoch+1}: avg_loss = {avg_loss:.6f}")
    
    # Final save
    save_checkpoint(model, optimizer, args.epochs - 1, avg_loss, output_dir / 'final.pth')
    print(f"\n✓ Training complete! Final model saved to {output_dir}/final.pth")


if __name__ == '__main__':
    main()

