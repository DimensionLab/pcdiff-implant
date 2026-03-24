"""
train_pcdiff.py — Self-contained PCDiff training for autoresearch.

THIS IS THE AGENT-EDITABLE FILE. The AI research agent modifies this file
to explore architectural and hyperparameter changes. Each experiment runs
for a fixed time budget, then evaluates using Chamfer Distance on a
validation subset.

The file must remain self-contained: all model definitions, diffusion
process, training loop, and evaluation call are in this single file.
It imports only from prepare_pcdiff.py (data + metrics) and standard libs.

Usage:
    python train_pcdiff.py                    # Train with defaults
    python train_pcdiff.py --time-budget 900  # 15 min budget
    python train_pcdiff.py --baseline         # Establish baseline metric
"""

import argparse
import functools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Add parent to path for pcdiff modules (needed for CUDA extensions)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pcdiff"))
from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule, Attention, Swish

# Autoresearch utilities
from prepare_pcdiff import (
    get_train_entries, get_eval_subset, evaluate_model,
    load_point_cloud, normalize_point_cloud, _nrrd_to_npy_path,
    NUM_POINTS, NUM_NN, SV_POINTS, RESULTS_DIR,
)

# ============================================================================
# HYPERPARAMETERS — Agent may modify these
# ============================================================================

# Diffusion process
BETA_START = 0.0001
BETA_END = 0.02
SCHEDULE_TYPE = "linear"     # "linear", "cosine", "warm0.1"
NUM_TIMESTEPS = 1000
LOSS_TYPE = "mse"
MODEL_MEAN_TYPE = "eps"      # "eps" (epsilon prediction)
MODEL_VAR_TYPE = "fixedsmall"

# Architecture
EMBED_DIM = 64
USE_ATTENTION = True
DROPOUT = 0.1
WIDTH_MULT = 1.0
VOX_RES_MULT = 1.0

# Training
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
BETA1 = 0.5
WEIGHT_DECAY = 0.0
GRAD_CLIP = None             # None or float (e.g., 1.0)
USE_AMP = False
AMP_DTYPE = "float16"        # "float16" or "bfloat16"

# LR schedule
LR_WARMUP_EPOCHS = 500
LR_WARMUP_START_FACTOR = 0.01
LR_GAMMA = 1.0              # ExponentialLR decay (1.0 = no decay)

# Data
AUGMENT = False              # Random rotation ±10°
NUM_WORKERS = 4

# Evaluation
EVAL_EVERY_EPOCHS = 50       # Run proxy eval every N epochs
DDIM_EVAL_STEPS = 50

# Time budget (seconds) — overridden by --time-budget flag
DEFAULT_TIME_BUDGET = 900    # 15 minutes

# ============================================================================
# NOISE SCHEDULE
# ============================================================================

def get_betas(schedule_type: str, b_start: float, b_end: float, num_steps: int) -> np.ndarray:
    """Generate beta schedule for diffusion process."""
    if schedule_type == "linear":
        return np.linspace(b_start, b_end, num_steps)
    elif schedule_type == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        steps = np.arange(num_steps + 1, dtype=np.float64)
        alpha_bar = np.cos(((steps / num_steps) + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return np.clip(betas, a_min=1e-6, a_max=0.999)
    elif schedule_type.startswith("warm"):
        warmup_frac = float(schedule_type.replace("warm", ""))
        betas = b_end * np.ones(num_steps, dtype=np.float64)
        warmup_time = int(num_steps * warmup_frac)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
        return betas
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

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
        sa_block_list = []
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
                    sa_block_list.append(block(in_channels, out_channels))
                elif k == 0:
                    sa_block_list.append(block(in_channels + embed_dim, out_channels))
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
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_block_list.append(
            block(in_channels=extra_feature_channels + (embed_dim if k == 0 else 0), out_channels=out_channels,
                  include_coordinates=True))
        c += 1
        in_channels = extra_feature_channels = sa_block_list[-1].out_channels
        if len(sa_block_list) == 1:
            sa_layers.append(sa_block_list[0])
        else:
            sa_layers.append(nn.Sequential(*sa_block_list))
    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, sv_points, embed_dim=64, use_att=False,
                                dropout=0.1, with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    fp_layers = []
    c = 0
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_block_list = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_block_list.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx] + embed_dim,
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
                fp_block_list.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_block_list) == 1:
            fp_layers.append(fp_block_list[0])
        else:
            fp_layers.append(nn.Sequential(*fp_block_list))
        c += 1
    return fp_layers, in_channels


class PVCNN2Base(nn.Module):
    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout=0.1,
                 extra_feature_channels=3, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        assert extra_feature_channels >= 0
        self.embed_dim = embed_dim
        self.sv_points = sv_points
        self.in_channels = extra_feature_channels + 3

        sa_layers, sa_in_channels, channels_sa_features, _ = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks, extra_feature_channels=extra_feature_channels, with_se=True,
            embed_dim=embed_dim, use_att=use_att, dropout=dropout,
            width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)
        self.sa_layers = nn.ModuleList(sa_layers)
        self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks, in_channels=channels_sa_features, sa_in_channels=sa_in_channels,
            sv_points=sv_points, with_se=True, embed_dim=embed_dim, use_att=use_att, dropout=dropout,
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
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
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
            features, coords, temb = fp_blocks(
                (jump_coords, coords, torch.cat([features, temb], dim=1), fump_feats, temb))

        return self.classifier(features)


class PVCNN2(PVCNN2Base):
    num_n = 128

    sa_blocks = [
        ((32, 2, 32), (10240, 0.1, num_n, (32, 64))),
        ((64, 3, 16), (2560, 0.2, num_n, (64, 128))),
        ((128, 3, 8), (640, 0.4, num_n, (128, 256))),
        (None, (160, 0.8, num_n, (256, 256, 512))),
    ]

    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout,
                 extra_feature_channels=3, width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


# ============================================================================
# GAUSSIAN DIFFUSION
# ============================================================================

class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.np_betas = betas = betas.astype(np.float64)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1.0, alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

        betas_t = torch.from_numpy(betas).float()
        alphas_t = torch.from_numpy(alphas).float()
        posterior_variance = betas_t * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas_t * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        bs, = t.shape
        out = torch.gather(a, 0, t)
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        B, D, N = data_start.shape
        if noise is None:
            noise = torch.randn(data_start[:, :, self.sv_points:].shape, device=data_start.device)

        x_noisy = torch.cat([
            data_start[:, :, :self.sv_points],
            self.q_sample(x_start=data_start[:, :, self.sv_points:], t=t, noise=noise)
        ], dim=-1)

        x_recon = denoise_fn(x_noisy, t)[:, :, self.sv_points:]

        if self.loss_type == 'mse':
            losses = ((noise - x_recon) ** 2).mean(dim=[1, 2])
        else:
            raise NotImplementedError(self.loss_type)
        return losses

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised=False, return_pred_xstart=True):
        model_output = denoise_fn(data, t)[:, :, self.sv_points:]

        if self.model_var_type == 'fixedsmall':
            model_variance = self._extract(self.posterior_variance.to(data.device), t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(self.posterior_log_variance_clipped.to(data.device), t, data.shape) * torch.ones_like(model_output)
        elif self.model_var_type == 'fixedlarge':
            model_variance = self._extract(self.betas.to(data.device), t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(
                torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device), t, data.shape
            ) * torch.ones_like(model_output)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:, :, self.sv_points:], t=t, eps=model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:, :, self.sv_points:], t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        return model_mean, model_variance, model_log_variance

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False):
        model_mean, _, model_log_variance, _ = self.p_mean_variance(denoise_fn, data=data, t=t,
                                                                     clip_denoised=clip_denoised, return_pred_xstart=True)
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return torch.cat([data[:, :, :self.sv_points], sample], dim=-1)

    def p_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True):
        noise = noise_fn(size=shape, dtype=torch.float, device=device)
        img_t = torch.cat([partial_x, noise], dim=-1)
        for t in reversed(range(0, self.num_timesteps)):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn, clip_denoised=clip_denoised)
        return img_t

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - pred_xstart
                ) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    def ddim_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=True, eta=0.0):
        _, _, _, x_start = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
        eps = self._predict_eps_from_xstart(data[:, :, self.sv_points:], t, x_start)
        alpha_bar = self._extract(self.alphas_cumprod.to(data.device), t, data.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev.to(data.device), t, data.shape)
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        noise = noise_fn(size=x_start.shape, dtype=x_start.dtype, device=x_start.device)
        mean_pred = x_start * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(data.shape) - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return torch.cat([data[:, :, :self.sv_points], sample], dim=-1)

    def ddim_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn,
                         clip_denoised=True, sampling_steps=50):
        noise = noise_fn(size=shape, dtype=torch.float, device=device)
        img_t = torch.cat([partial_x, noise], dim=-1)

        step_size = self.num_timesteps // sampling_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        for i, t_val in enumerate(timesteps):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t_val)
            img_t = self.ddim_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                     clip_denoised=clip_denoised)
        return img_t


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class Model(nn.Module):
    def __init__(self, betas):
        super().__init__()
        self.diffusion = GaussianDiffusion(betas, LOSS_TYPE, MODEL_MEAN_TYPE, MODEL_VAR_TYPE, sv_points=SV_POINTS)
        self.model = PVCNN2(num_classes=3, sv_points=SV_POINTS, embed_dim=EMBED_DIM,
                            use_att=USE_ATTENTION, dropout=DROPOUT, extra_feature_channels=0,
                            width_multiplier=WIDTH_MULT, voxel_resolution_multiplier=VOX_RES_MULT)

    def _denoise(self, data, t):
        return self.model(data, t)

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)
        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)
        losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, noise=noises)
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn, clip_denoised=True,
                    sampling_method='ddpm', sampling_steps=1000):
        if sampling_method == 'ddim':
            return self.diffusion.ddim_sample_loop(partial_x, self._denoise, shape=shape, device=device,
                                                   noise_fn=noise_fn, clip_denoised=clip_denoised,
                                                   sampling_steps=sampling_steps)
        else:
            return self.diffusion.p_sample_loop(partial_x, self._denoise, shape=shape, device=device,
                                                noise_fn=noise_fn, clip_denoised=clip_denoised)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ============================================================================
# DATASET
# ============================================================================

class SkullBreakDatasetSimple(torch.utils.data.Dataset):
    """Simplified SkullBreak dataset for autoresearch training."""

    def __init__(self, entries, augment=False):
        self.entries = entries
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        defective_path, implant_path = self.entries[idx]

        # Load and subsample
        pc_defective = load_point_cloud(defective_path, SV_POINTS)
        pc_implant = load_point_cloud(implant_path, NUM_NN)
        pc_combined = np.concatenate([pc_defective, pc_implant], axis=0)

        if self.augment:
            # Random rotation ±10° around all axes
            angles = np.random.uniform(-10, 10, size=3) * np.pi / 180
            cx, sx = np.cos(angles[0]), np.sin(angles[0])
            cy, sy = np.cos(angles[1]), np.sin(angles[1])
            cz, sz = np.cos(angles[2]), np.sin(angles[2])
            Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx
            center = pc_combined.mean(axis=0)
            pc_combined = (pc_combined - center) @ R.T + center

        # Normalize
        pc_normalized, _, _ = normalize_point_cloud(pc_combined)
        return {
            'idx': idx,
            'train_points': torch.from_numpy(pc_normalized).float(),
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_with_budget(time_budget: int, checkpoint_path: str = None, baseline: bool = False):
    """
    Train PCDiff model within a fixed time budget.

    Returns:
        dict with final metrics, epochs trained, and config summary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Time budget: {time_budget}s ({time_budget / 60:.1f} min)")

    # Create model
    betas = get_betas(SCHEDULE_TYPE, BETA_START, BETA_END, NUM_TIMESTEPS)
    model = Model(betas)
    model = model.cuda() if torch.cuda.is_available() else model

    # Load checkpoint if provided
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999), weight_decay=WEIGHT_DECAY)
    if checkpoint_path and os.path.exists(checkpoint_path):
        if 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])

    # LR scheduler
    if LR_WARMUP_EPOCHS > 0:
        warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=LR_WARMUP_START_FACTOR,
                                              end_factor=1.0, total_iters=LR_WARMUP_EPOCHS)
        main_sched = optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)
        lr_scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, main_sched],
                                                       milestones=[LR_WARMUP_EPOCHS])
    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, LR_GAMMA)

    # Fast-forward scheduler to current epoch
    for _ in range(start_epoch):
        lr_scheduler.step()

    # AMP
    amp_dtype = torch.bfloat16 if AMP_DTYPE == "bfloat16" else torch.float16
    use_amp = USE_AMP and torch.cuda.is_available()
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    # Dataset
    train_entries = get_train_entries()
    dataset = SkullBreakDatasetSimple(train_entries, augment=AUGMENT)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, drop_last=True, pin_memory=True,
    )

    # Init noise
    noises_init = torch.randn(len(dataset), NUM_NN, 3)

    # Eval entries (fixed)
    eval_entries = get_eval_subset()

    print(f"Training samples: {len(dataset)}, Batch size: {BATCH_SIZE}, Batches/epoch: {len(dataloader)}")
    print(f"Config: embed_dim={EMBED_DIM}, attention={USE_ATTENTION}, dropout={DROPOUT}")
    print(f"         width_mult={WIDTH_MULT}, vox_res_mult={VOX_RES_MULT}")
    print(f"         schedule={SCHEDULE_TYPE}, timesteps={NUM_TIMESTEPS}")
    print(f"         lr={LEARNING_RATE}, beta1={BETA1}, grad_clip={GRAD_CLIP}")

    # If baseline mode, just evaluate current model
    if baseline:
        print("\n=== BASELINE EVALUATION ===")
        metrics = evaluate_model(model, device, eval_entries, DDIM_EVAL_STEPS)
        print(f"Chamfer Distance: {metrics['chamfer_mean']:.6f} ± {metrics['chamfer_std']:.6f}")
        print(f"Eval time: {metrics['eval_time_sec']:.1f}s")
        return {"metrics": metrics, "epochs": 0, "baseline": True}

    # Training loop with time budget
    t_start = time.time()
    epoch = start_epoch
    best_loss = float("inf")
    loss_history = []

    model.train()
    print(f"\n=== TRAINING START (epoch {epoch}) ===")

    while True:
        elapsed = time.time() - t_start
        # Reserve 3 minutes for final evaluation
        if elapsed > (time_budget - 180):
            print(f"\nTime budget reached at epoch {epoch} ({elapsed:.0f}s elapsed)")
            break

        epoch_losses = []
        for i, data in enumerate(dataloader):
            pc_in = data['train_points'].transpose(1, 2)  # [B, 3, N]
            noises_batch = noises_init[data['idx']].transpose(1, 2)

            if torch.cuda.is_available():
                pc_in = pc_in.cuda(non_blocking=True)
                noises_batch = noises_batch.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                loss = model.get_loss_iter(pc_in, noises_batch).mean()

            loss_val = loss.item()
            epoch_losses.append(loss_val)

            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
                if GRAD_CLIP is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            # Check time within epoch
            if time.time() - t_start > (time_budget - 180):
                break

        lr_scheduler.step()
        epoch_mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        loss_history.append(epoch_mean_loss)

        if epoch_mean_loss < best_loss:
            best_loss = epoch_mean_loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d} | Loss: {epoch_mean_loss:.6f} | Best: {best_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | {time.time() - t_start:.0f}s")

        # Periodic proxy evaluation
        if epoch > 0 and epoch % EVAL_EVERY_EPOCHS == 0:
            metrics = evaluate_model(model, device, eval_entries, DDIM_EVAL_STEPS)
            print(f"  >> Proxy eval: CD={metrics['chamfer_mean']:.6f} ± {metrics['chamfer_std']:.6f} "
                  f"({metrics['eval_time_sec']:.1f}s)")
            model.train()

        # NaN detection
        if np.isnan(epoch_mean_loss):
            print("ERROR: NaN loss detected. Stopping.")
            break

        epoch += 1

    # Final evaluation
    print(f"\n=== FINAL EVALUATION (after {epoch - start_epoch} epochs) ===")
    metrics = evaluate_model(model, device, eval_entries, DDIM_EVAL_STEPS)
    print(f"Chamfer Distance: {metrics['chamfer_mean']:.6f} ± {metrics['chamfer_std']:.6f}")
    print(f"Eval time: {metrics['eval_time_sec']:.1f}s")
    print(f"Total training time: {time.time() - t_start:.1f}s")

    # Save checkpoint
    ckpt_dir = RESULTS_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pth"
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'loss': best_loss,
        'metrics': metrics,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    # Save best if this is the best run
    best_metric_file = RESULTS_DIR / "best_chamfer.txt"
    prev_best = float("inf")
    if best_metric_file.exists():
        prev_best = float(best_metric_file.read_text().strip())

    if metrics["chamfer_mean"] < prev_best:
        best_path = ckpt_dir / "best.pth"
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'loss': best_loss,
            'metrics': metrics,
        }, best_path)
        best_metric_file.write_text(str(metrics["chamfer_mean"]))
        print(f"NEW BEST! Saved to {best_path}")

    return {
        "metrics": metrics,
        "epochs_trained": epoch - start_epoch,
        "total_epochs": epoch,
        "best_loss": best_loss,
        "final_loss": loss_history[-1] if loss_history else float("nan"),
        "config": {
            "schedule_type": SCHEDULE_TYPE,
            "num_timesteps": NUM_TIMESTEPS,
            "embed_dim": EMBED_DIM,
            "use_attention": USE_ATTENTION,
            "dropout": DROPOUT,
            "width_mult": WIDTH_MULT,
            "vox_res_mult": VOX_RES_MULT,
            "learning_rate": LEARNING_RATE,
            "beta1": BETA1,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE,
            "augment": AUGMENT,
            "use_amp": USE_AMP,
        },
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCDiff autoresearch training")
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET,
                        help=f"Training time budget in seconds (default: {DEFAULT_TIME_BUDGET})")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--baseline", action="store_true", help="Only run evaluation (no training)")
    args = parser.parse_args()

    result = train_with_budget(args.time_budget, args.checkpoint, args.baseline)
    print(f"\n=== RESULT ===")
    print(json.dumps(result, indent=2, default=str))
