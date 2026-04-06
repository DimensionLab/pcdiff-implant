import argparse
import datetime
import hashlib
import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.amp import GradScaler, autocast
from torch.distributions import Normal

from datasets.skullbreak_data import SkullBreakDataset
from datasets.skullfix_data import SkullFixDataset
from model.pvcnn_completion import PVCNN2Base
from utils.file_utils import copy_source, get_output_dir, setup_logging, setup_output_subdirs
from utils.visualize import export_to_pc_batch
import glob
import subprocess
from pathlib import Path

# Import proxy evaluation module (optional - may fail if dependencies not installed)
try:
    from proxy_eval import run_proxy_evaluation, save_proxy_metrics, VOXELIZATION_AVAILABLE
    PROXY_EVAL_AVAILABLE = VOXELIZATION_AVAILABLE
except ImportError:
    PROXY_EVAL_AVAILABLE = False

# Optional wandb integration for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class StopReason(Enum):
    """Reasons for stopping training in the gating loop."""
    CONTINUE = "continue"
    MAX_EPOCHS = "max_epochs_reached"
    NAN_INF = "nan_inf_detected"
    EXPLODING_GRAD = "exploding_gradients"
    LOSS_DIVERGENCE = "loss_divergence"
    PLATEAU = "plateau_detected"
    USER_STOP = "user_requested_stop"


@dataclass
class GatingConfig:
    """Configuration for the 700-epoch gating loop."""
    # Decision checkpoints
    decision_epochs: List[int] = field(default_factory=lambda: [50, 100, 200, 500, 700])
    max_epochs: int = 700

    # Proxy eval frequency
    proxy_eval_freq: int = 200

    # Divergence detection thresholds
    nan_check: bool = True
    grad_norm_threshold: float = 1e6  # Exploding gradient threshold
    loss_spike_threshold: float = 10.0  # Loss spike factor vs running median

    # Plateau detection parameters
    plateau_delta_threshold: float = 0.005  # Minimum improvement in proxy metrics
    plateau_loss_variance_threshold: float = 0.3  # High variance threshold (90p - 10p) / median
    plateau_consecutive_checks: int = 2  # Consecutive plateau checks before stopping

    # Enable/disable gating (useful for debugging)
    enabled: bool = True


@dataclass
class EpochStats:
    """Statistics collected during an epoch."""
    epoch: int
    loss_values: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    lr: float = 0.0

    @property
    def loss_mean(self) -> float:
        return np.mean(self.loss_values) if self.loss_values else float('nan')

    @property
    def loss_median(self) -> float:
        return np.median(self.loss_values) if self.loss_values else float('nan')

    @property
    def loss_std(self) -> float:
        return np.std(self.loss_values) if len(self.loss_values) > 1 else 0.0

    @property
    def grad_norm_mean(self) -> float:
        return np.mean(self.grad_norms) if self.grad_norms else 0.0

    @property
    def grad_norm_max(self) -> float:
        return np.max(self.grad_norms) if self.grad_norms else 0.0


@dataclass
class ProxyMetrics:
    """Proxy evaluation metrics (placeholder for actual metrics from inference)."""
    epoch: int
    dsc: float = 0.0
    bdsc: float = 0.0
    hd95: float = float('inf')

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "dsc": self.dsc,
            "bdsc": self.bdsc,
            "hd95": self.hd95
        }


class GatingLoopTracker:
    """
    Tracks training statistics and makes continue/stop decisions at gating checkpoints.

    Implements the 700-epoch gating loop with:
    - Decision checkpoints at 50/100/200/500/700 epochs
    - Stop-on-divergence: NaN/Inf, exploding gradients, loss spike
    - Stop-on-plateau: No proxy metric improvement + high-variance loss
    """

    def __init__(self, config: GatingConfig, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Epoch statistics history (keeps all epochs for analysis)
        self.epoch_stats: List[EpochStats] = []

        # Proxy metrics history
        self.proxy_metrics: List[ProxyMetrics] = []

        # Plateau tracking
        self.consecutive_plateau_count = 0
        self.last_decision_epoch = 0

        # Stop state
        self.should_stop = False
        self.stop_reason = StopReason.CONTINUE
        self.stop_details = ""

    def record_epoch(self, stats: EpochStats) -> None:
        """Record statistics for an epoch."""
        self.epoch_stats.append(stats)

    def record_proxy_metrics(self, metrics: ProxyMetrics) -> None:
        """Record proxy evaluation metrics."""
        self.proxy_metrics.append(metrics)

    def check_nan_inf(self, loss: float, grad_norm: float) -> Tuple[bool, str]:
        """Check for NaN/Inf in loss or gradients."""
        if not self.config.nan_check:
            return False, ""

        if np.isnan(loss) or np.isinf(loss):
            return True, f"NaN/Inf loss detected: {loss}"

        if np.isnan(grad_norm) or np.isinf(grad_norm):
            return True, f"NaN/Inf gradient norm detected: {grad_norm}"

        return False, ""

    def check_exploding_gradients(self, grad_norm: float) -> Tuple[bool, str]:
        """Check for exploding gradients."""
        if grad_norm > self.config.grad_norm_threshold:
            return True, f"Exploding gradients: grad_norm={grad_norm:.2e} > threshold={self.config.grad_norm_threshold:.2e}"
        return False, ""

    def check_loss_spike(self, loss: float) -> Tuple[bool, str]:
        """Check for catastrophic loss spike."""
        if len(self.epoch_stats) < 10:
            return False, ""

        # Get running median of last 50 epochs
        recent_losses = []
        for stats in self.epoch_stats[-50:]:
            recent_losses.extend(stats.loss_values)

        if not recent_losses:
            return False, ""

        running_median = np.median(recent_losses)
        if running_median > 0 and loss > running_median * self.config.loss_spike_threshold:
            return True, f"Catastrophic loss spike: {loss:.4f} > {self.config.loss_spike_threshold}x median ({running_median:.4f})"

        return False, ""

    def get_loss_summary(self, last_n_epochs: int = 50) -> Dict[str, float]:
        """Get loss summary statistics over the last N epochs."""
        if not self.epoch_stats:
            return {"median": float('nan'), "p10": float('nan'), "p90": float('nan'), "std": float('nan')}

        recent_stats = self.epoch_stats[-last_n_epochs:]
        all_losses = []
        for stats in recent_stats:
            all_losses.extend(stats.loss_values)

        if not all_losses:
            return {"median": float('nan'), "p10": float('nan'), "p90": float('nan'), "std": float('nan')}

        return {
            "median": float(np.median(all_losses)),
            "p10": float(np.percentile(all_losses, 10)),
            "p90": float(np.percentile(all_losses, 90)),
            "std": float(np.std(all_losses)),
            "mean": float(np.mean(all_losses)),
        }

    def check_plateau(self, current_epoch: int) -> Tuple[bool, str]:
        """
        Check for training plateau.

        Plateau is detected when:
        1. Proxy metrics improve by less than delta_threshold over the last 100 epochs AND
        2. Loss is in a high-variance band without downward trend
        """
        # Need at least 2 proxy evals to compare
        if len(self.proxy_metrics) < 2:
            return False, ""

        # Check proxy metric improvement over last 100 epochs
        # Find metrics from ~100 epochs ago
        current_proxy = self.proxy_metrics[-1]
        old_proxy = None
        for m in reversed(self.proxy_metrics[:-1]):
            if current_proxy.epoch - m.epoch >= 100:
                old_proxy = m
                break

        if old_proxy is None and len(self.proxy_metrics) >= 2:
            old_proxy = self.proxy_metrics[-2]

        if old_proxy is None:
            return False, ""

        # Calculate improvements
        dsc_improvement = current_proxy.dsc - old_proxy.dsc
        bdsc_improvement = current_proxy.bdsc - old_proxy.bdsc

        # Check if improvements are below threshold
        metrics_stagnant = (
            dsc_improvement < self.config.plateau_delta_threshold and
            bdsc_improvement < self.config.plateau_delta_threshold
        )

        if not metrics_stagnant:
            self.consecutive_plateau_count = 0
            return False, ""

        # Check loss variance
        loss_summary = self.get_loss_summary(50)
        loss_median = loss_summary["median"]
        loss_spread = loss_summary["p90"] - loss_summary["p10"]

        if loss_median > 0:
            relative_spread = loss_spread / loss_median
            high_variance = relative_spread > self.config.plateau_loss_variance_threshold
        else:
            high_variance = False

        if metrics_stagnant and high_variance:
            self.consecutive_plateau_count += 1
            if self.consecutive_plateau_count >= self.config.plateau_consecutive_checks:
                return True, (
                    f"Plateau detected: DSC improvement={dsc_improvement:.4f}, "
                    f"bDSC improvement={bdsc_improvement:.4f} (threshold={self.config.plateau_delta_threshold}), "
                    f"loss variance={(relative_spread*100):.1f}% (threshold={self.config.plateau_loss_variance_threshold*100}%), "
                    f"consecutive checks={self.consecutive_plateau_count}"
                )

        return False, ""

    def check_loss_divergence(self) -> Tuple[bool, str]:
        """
        Check for loss divergence: median increases for 2 consecutive checkpoints
        AND 90-percentile spikes worsen.
        """
        if len(self.epoch_stats) < 100:
            return False, ""

        # Compare loss summaries at current vs 50 epochs ago vs 100 epochs ago
        current_summary = self.get_loss_summary(25)
        mid_summary = self._get_loss_summary_at_range(-75, -50)
        old_summary = self._get_loss_summary_at_range(-100, -75)

        if any(np.isnan(s.get("median", float('nan'))) for s in [current_summary, mid_summary, old_summary]):
            return False, ""

        # Check if median increased for 2 consecutive periods
        median_increasing = (
            current_summary["median"] > mid_summary["median"] > old_summary["median"]
        )

        # Check if 90th percentile spikes worsened
        p90_worsening = (
            current_summary["p90"] > mid_summary["p90"] > old_summary["p90"]
        )

        if median_increasing and p90_worsening:
            return True, (
                f"Loss divergence: median {old_summary['median']:.4f} -> {mid_summary['median']:.4f} -> {current_summary['median']:.4f}, "
                f"p90 {old_summary['p90']:.4f} -> {mid_summary['p90']:.4f} -> {current_summary['p90']:.4f}"
            )

        return False, ""

    def _get_loss_summary_at_range(self, start_offset: int, end_offset: int) -> Dict[str, float]:
        """Get loss summary for a range of epochs relative to current."""
        if not self.epoch_stats:
            return {"median": float('nan'), "p10": float('nan'), "p90": float('nan')}

        total = len(self.epoch_stats)
        start_idx = max(0, total + start_offset)
        end_idx = max(0, total + end_offset)

        if start_idx >= end_idx:
            return {"median": float('nan'), "p10": float('nan'), "p90": float('nan')}

        all_losses = []
        for stats in self.epoch_stats[start_idx:end_idx]:
            all_losses.extend(stats.loss_values)

        if not all_losses:
            return {"median": float('nan'), "p10": float('nan'), "p90": float('nan')}

        return {
            "median": float(np.median(all_losses)),
            "p10": float(np.percentile(all_losses, 10)),
            "p90": float(np.percentile(all_losses, 90)),
        }

    def is_decision_epoch(self, epoch: int) -> bool:
        """Check if current epoch is a decision checkpoint."""
        return epoch in self.config.decision_epochs

    def is_proxy_eval_epoch(self, epoch: int) -> bool:
        """Check if current epoch should run proxy evaluation."""
        return epoch > 0 and (epoch % self.config.proxy_eval_freq == 0)

    def evaluate_gating_decision(self, epoch: int) -> Tuple[StopReason, str]:
        """
        Make a continue/stop decision at a gating checkpoint.

        Returns:
            (StopReason, details_string)
        """
        if not self.config.enabled:
            return StopReason.CONTINUE, "Gating disabled"

        # Check max epochs
        if epoch >= self.config.max_epochs:
            return StopReason.MAX_EPOCHS, f"Reached max epochs ({self.config.max_epochs})"

        # Check loss divergence
        is_diverging, details = self.check_loss_divergence()
        if is_diverging:
            return StopReason.LOSS_DIVERGENCE, details

        # Check plateau (only if we have proxy metrics)
        is_plateau, details = self.check_plateau(epoch)
        if is_plateau:
            return StopReason.PLATEAU, details

        self.last_decision_epoch = epoch
        return StopReason.CONTINUE, f"Continuing to next checkpoint"

    def step_check(self, loss: float, grad_norm: float) -> Tuple[bool, StopReason, str]:
        """
        Per-step check for immediate stopping conditions (NaN, exploding gradients).

        Returns:
            (should_stop, reason, details)
        """
        if not self.config.enabled:
            return False, StopReason.CONTINUE, ""

        # Check NaN/Inf
        is_nan, details = self.check_nan_inf(loss, grad_norm)
        if is_nan:
            self.should_stop = True
            self.stop_reason = StopReason.NAN_INF
            self.stop_details = details
            return True, StopReason.NAN_INF, details

        # Check exploding gradients
        is_exploding, details = self.check_exploding_gradients(grad_norm)
        if is_exploding:
            self.should_stop = True
            self.stop_reason = StopReason.EXPLODING_GRAD
            self.stop_details = details
            return True, StopReason.EXPLODING_GRAD, details

        # Check loss spike
        is_spike, details = self.check_loss_spike(loss)
        if is_spike:
            self.should_stop = True
            self.stop_reason = StopReason.LOSS_DIVERGENCE
            self.stop_details = details
            return True, StopReason.LOSS_DIVERGENCE, details

        return False, StopReason.CONTINUE, ""

    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training statistics."""
        loss_summary = self.get_loss_summary(50)

        summary = {
            "total_epochs": len(self.epoch_stats),
            "last_epoch": self.epoch_stats[-1].epoch if self.epoch_stats else 0,
            "loss_summary_last_50": loss_summary,
            "stop_reason": self.stop_reason.value,
            "stop_details": self.stop_details,
            "proxy_metrics_count": len(self.proxy_metrics),
        }

        if self.proxy_metrics:
            latest = self.proxy_metrics[-1]
            summary["latest_proxy_metrics"] = latest.to_dict()

            # Best metrics
            best_dsc = max(m.dsc for m in self.proxy_metrics)
            best_bdsc = max(m.bdsc for m in self.proxy_metrics)
            best_hd95 = min(m.hd95 for m in self.proxy_metrics)
            summary["best_proxy_metrics"] = {
                "best_dsc": best_dsc,
                "best_bdsc": best_bdsc,
                "best_hd95": best_hd95,
            }

        return summary

    def save_state(self, path: str) -> None:
        """Save tracker state to JSON file."""
        state = {
            "config": {
                "decision_epochs": self.config.decision_epochs,
                "max_epochs": self.config.max_epochs,
                "proxy_eval_freq": self.config.proxy_eval_freq,
                "enabled": self.config.enabled,
            },
            "epoch_count": len(self.epoch_stats),
            "proxy_metrics": [m.to_dict() for m in self.proxy_metrics],
            "consecutive_plateau_count": self.consecutive_plateau_count,
            "should_stop": self.should_stop,
            "stop_reason": self.stop_reason.value,
            "stop_details": self.stop_details,
            "training_summary": self.get_training_summary(),
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)


'''
----- Some utilities -----
'''


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotate(vertices, faces):
    """ vertices: [numpoints, 3] """
    M = rotation_matrix([0, 1, 0], np.pi / 2).transpose()
    N = rotation_matrix([1, 0, 0], -np.pi / 4).transpose()
    K = rotation_matrix([0, 0, 1], np.pi).transpose()

    v, f = vertices[:, [1, 2, 0]].dot(M).dot(N).dot(K), faces[:, [1, 2, 0]]
    return v, f


def norm(v, f):
    v = (v - v.min()) / (v.max() - v.min()) - 0.5

    return v, f


def getGradNorm(net):
    pNorm = torch.sqrt(sum(torch.sum(p ** 2) for p in net.parameters()))
    gradNorm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in net.parameters()))

    return pNorm, gradNorm


def weights_init(m):
    """
    xavier initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and m.weight is not None:
        torch.nn.init.xavier_normal_(m.weight)

    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_()
        m.bias.data.fill_(0)


''' 
----- Models ----- 
'''


def seed_everything(seed: int, deterministic: bool = True) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_distributed_context():
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return world_size, rank, local_rank


class CheckpointManager:
    """Manages model checkpoints: keeps best model and last N periodic checkpoints."""

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3, logger=None):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        self.logger = logger
        self.best_loss = float('inf')
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, epoch: int, loss: float, is_periodic: bool = False) -> Dict[str, Any]:
        """
        Save checkpoint and manage old ones.
        Returns a dict describing what was saved and where, so callers can attach external logging (e.g., W&B artifacts).
        """
        save_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'loss': loss,
        }

        # Always save latest
        latest_path = os.path.join(self.checkpoint_dir, 'model_latest.pth')
        torch.save(save_dict, latest_path)
        saved_best = False
        best_path = None

        # Save best model if this is the best loss
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(save_dict, best_path)
            saved_best = True
            if self.logger:
                self.logger.info(f'New best model saved at epoch {epoch} with loss {loss:.6f}')

        # Save periodic checkpoint
        if is_periodic:
            epoch_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pth')
            torch.save(save_dict, epoch_path)
            if self.logger:
                self.logger.info(f'Periodic checkpoint saved: {epoch_path}')

            # Clean up old periodic checkpoints
            self._cleanup_old_checkpoints()
        else:
            epoch_path = None

        return {
            "latest_path": latest_path,
            "best_path": best_path,
            "saved_best": saved_best,
            "periodic_path": epoch_path,
            "saved_periodic": bool(epoch_path),
        }

    def _cleanup_old_checkpoints(self):
        """Keep only the last N periodic checkpoints."""
        pattern = os.path.join(self.checkpoint_dir, 'model_epoch_*.pth')
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)

        # Remove oldest checkpoints if we have more than keep_last_n
        while len(checkpoints) > self.keep_last_n:
            old_ckpt = checkpoints.pop(0)
            os.remove(old_ckpt)
            if self.logger:
                self.logger.info(f'Removed old checkpoint: {old_ckpt}')

    def load_best_loss(self, checkpoint_path: str):
        """Load best loss from a checkpoint if resuming."""
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if 'loss' in ckpt:
                self.best_loss = ckpt['loss']


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains shadow copies of model parameters that are updated as an
    exponential moving average during training.  Call ``apply_shadow`` before
    evaluation to swap in the averaged weights and ``restore`` afterwards to
    resume training with the original parameters.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """Update shadow parameters with current model parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        """Swap model parameters with shadow (EMA) parameters for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore original model parameters after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + (mean1 - mean2) ** 2 * torch.exp(-logvar2))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    # Assumes data is integers [0, 1]
    assert x.shape == means.shape == log_scales.shape
    px0 = Normal(torch.zeros_like(means), torch.ones_like(log_scales))

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 0.5)
    cdf_plus = px0.cdf(plus_in)
    min_in = inv_stdv * (centered_x - .5)
    cdf_min = px0.cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.ones_like(cdf_plus) * 1e-12))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.ones_like(cdf_min) * 1e-12))
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < 0.001, log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min,
                    torch.log(torch.max(cdf_delta, torch.ones_like(cdf_delta) * 1e-12))))
    assert log_probs.shape == x.shape
    return log_probs


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type, sv_points, min_snr_gamma=5.0):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self._min_snr_gamma = min_snr_gamma
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sv_points = sv_points
        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1. - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(np.append(1., alphas_cumprod[:-1])).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance)))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)

        # SNR for Min-SNR-γ weighting (used by 'mse_minsnr' loss type)
        self.snr = alphas_cumprod / (1.0 - alphas_cumprod)

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        bs, = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])

        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start
        variance = self._extract(1. - self.alphas_cumprod.to(x_start.device), t, x_start.shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape)

        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """ Diffuse the data (t == 0 means diffused for 1 step) """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)

        assert noise.shape == x_start.shape

        return (self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape) * x_start +
                self._extract(self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape) * noise)

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """ Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0) """
        assert x_start.shape == x_t.shape
        posterior_mean = (self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape) * x_start +
                          self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance.to(x_start.device), t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped.to(x_start.device), t,
                                                       x_t.shape)
        assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
                x_start.shape[0])

        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool):

        model_output = denoise_fn(data, t)[:, :, self.sv_points:]

        if self.model_var_type in ['fixedsmall', 'fixedlarge']:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas.to(data.device),
                               torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(data.device)),
                'fixedsmall': (self.posterior_variance.to(data.device),
                               self.posterior_log_variance_clipped.to(data.device))}[self.model_var_type]

            model_variance = self._extract(model_variance, t, data.shape) * torch.ones_like(model_output)
            model_log_variance = self._extract(model_log_variance, t, data.shape) * torch.ones_like(model_output)

        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == 'eps':
            x_recon = self._predict_xstart_from_eps(data[:, :, self.sv_points:], t=t, eps=model_output)

            model_mean, _, _ = self.q_posterior_mean_variance(x_start=x_recon, x_t=data[:, :, self.sv_points:], t=t)

        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape
        assert model_variance.shape == model_log_variance.shape

        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon

        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t -
                self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape) * eps)

    ''' 
    ----- Sampling ----- 
    '''

    def p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
        """ Sample from the model """
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(denoise_fn, data=data, t=t,
                                                                              clip_denoised=clip_denoised,
                                                                              return_pred_xstart=True)
        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)

        # no noise when t == 0
        nonzero_mask = torch.reshape(1 - (t == 0).float(), [data.shape[0]] + [1] * (len(model_mean.shape) - 1))

        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                      keep_running=False):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """

        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)
        for t in reversed(range(0, self.num_timesteps if not keep_running else len(self.betas))):
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    '''
    ----- DDIM Sampling -----
    '''

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """Predict epsilon from x_start and x_t."""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape) * x_t - pred_xstart
        ) / self._extract(self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape)

    def ddim_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=True, return_pred_xstart=True, eta=0.0):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        model_mean, _, _, x_start = self.p_mean_variance(denoise_fn, data=data, t=t, clip_denoised=clip_denoised,
                                                         return_pred_xstart=return_pred_xstart)

        eps = self._predict_eps_from_xstart(data[:, :, self.sv_points:], t, x_start)

        alpha_bar = self._extract(self.alphas_cumprod.to(data.device), t, data.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod_prev.to(data.device), t, data.shape)
        sigma = (eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        noise = noise_fn(size=model_mean.shape, dtype=model_mean.dtype, device=model_mean.device)
        mean_pred = (x_start * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(data.shape) - 1))))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        sample = torch.cat([data[:, :, :self.sv_points], sample], dim=-1)
        return sample

    def ddim_sample_loop(self, partial_x, denoise_fn, shape, device, noise_fn=torch.randn, clip_denoised=True,
                         sampling_steps=50):
        """
        Generate samples using DDIM (faster than DDPM).

        Args:
            partial_x: Partial point cloud (skull points)
            denoise_fn: Denoising function
            shape: Shape of noise to generate
            device: CUDA device
            noise_fn: Noise generation function
            clip_denoised: Whether to clip denoised values
            sampling_steps: Number of DDIM steps (default 50)
        """
        assert isinstance(shape, (tuple, list))
        noise = noise_fn(size=shape, dtype=torch.float, device=device)

        img_t = torch.cat([partial_x, noise], dim=-1)

        # Create timestep schedule: linspace from 0 to 999 with sampling_steps
        ts = np.linspace(0, 999, sampling_steps).round().astype('int')
        ts = np.unique(ts)[::-1]  # Remove duplicates and reverse

        for t in ts:
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.ddim_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                     clip_denoised=clip_denoised, return_pred_xstart=True)

        assert img_t[:, :, self.sv_points:].shape == shape
        return img_t

    def p_sample_loop_trajectory(self, denoise_fn, shape, device, freq, noise_fn=torch.randn, clip_denoised=True,
                                 keep_running=False):
        """
        Generate samples, returning intermediate images
        Useful for visualizing how denoised images evolve over time
        Args:
          repeat_noise_steps (int): Number of denoising timesteps in which the same noise
            is used across the batch. If >= 0, the initial noise is the same for all batch elemements.
        """
        assert isinstance(shape, (tuple, list))

        total_steps = self.num_timesteps if not keep_running else len(self.betas)

        img_t = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [img_t]
        for t in reversed(range(0, total_steps)):

            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(denoise_fn=denoise_fn, data=img_t, t=t_, noise_fn=noise_fn,
                                  clip_denoised=clip_denoised, return_pred_xstart=False)
            if t % freq == 0 or t == total_steps - 1:
                imgs.append(img_t)

        assert imgs[-1].shape == shape
        return imgs

    ''' 
    ----- Losses ----- 
    '''

    def _vb_terms_bpd(self, denoise_fn, data_start, data_t, t, clip_denoised: bool, return_pred_xstart: bool):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=data_start[:, :, self.sv_points:], x_t=data_t[:, :, self.sv_points:], t=t)
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn, data=data_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = kl.mean(dim=list(range(1, len(model_mean.shape)))) / np.log(2.)

        return (kl, pred_xstart) if return_pred_xstart else kl

    def p_losses(self, denoise_fn, data_start, t, noise=None):
        """ Training loss calculation """
        B, D, N = data_start.shape
        assert t.shape == torch.Size([B])

        if noise is None:
            noise = torch.randn(data_start[:, :, self.sv_points:].shape, dtype=data_start.dtype, device=data_start.device)

        # Diffuse masked points t times. Other points don't get diffused.
        data_t = self.q_sample(x_start=data_start[:, :, self.sv_points:], t=t, noise=noise)

        if self.loss_type == 'mse':
            # Predict the noise instead of x_start. Seems to be weighted naturally like SNR.
            # Apply network to estimate applied noise.
            eps_recon = denoise_fn(torch.cat([data_start[:, :, :self.sv_points], data_t], dim=-1), t)[:, :, self.sv_points:]

            # MSE between noise and predicted noise
            losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))

        elif self.loss_type == 'mse_minsnr':
            # Min-SNR-γ weighted MSE loss (Hang et al., "Efficient Diffusion Training via Min-SNR Weighting")
            # Reduces gradient variance across timesteps for faster convergence.
            eps_recon = denoise_fn(torch.cat([data_start[:, :, :self.sv_points], data_t], dim=-1), t)[:, :, self.sv_points:]
            raw_losses = ((noise - eps_recon) ** 2).mean(dim=list(range(1, len(data_start.shape))))

            # Min-SNR-γ weighting: weight = min(SNR(t), γ) / SNR(t)
            min_snr_gamma = getattr(self, '_min_snr_gamma', 5.0)
            snr_t = self._extract(self.snr.to(data_start.device), t, torch.Size([B, 1]))
            snr_t = snr_t.squeeze(-1)
            weights = torch.clamp(snr_t, max=min_snr_gamma) / snr_t
            losses = raw_losses * weights

        elif self.loss_type == 'kl':
            losses = self._vb_terms_bpd(
                denoise_fn=denoise_fn, data_start=data_start, data_t=data_t, t=t, clip_denoised=False,
                return_pred_xstart=False)
        else:
            raise NotImplementedError(self.loss_type)

        assert losses.shape == torch.Size([B])
        return losses

    ''' 
    ----- Debug ----- 
    '''

    def _prior_bpd(self, x_start):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps
            t_ = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(T - 1)
            qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=t_)
            kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance,
                                 mean2=torch.tensor([0.]).to(qt_mean), logvar2=torch.tensor([0.]).to(qt_log_variance))
            assert kl_prior.shape == x_start.shape
            return kl_prior.mean(dim=list(range(1, len(kl_prior.shape)))) / np.log(2.)

    def calc_bpd_loop(self, denoise_fn, x_start, clip_denoised=True):

        with torch.no_grad():
            B, T = x_start.shape[0], self.num_timesteps

            vals_bt_, mse_bt_ = torch.zeros([B, T], device=x_start.device), torch.zeros([B, T], device=x_start.device)
            for t in reversed(range(T)):
                t_b = torch.empty(B, dtype=torch.int64, device=x_start.device).fill_(t)
                # Calculate VLB term at the current timestep
                data_t = torch.cat(
                    [x_start[:, :, :self.sv_points], self.q_sample(x_start=x_start[:, :, self.sv_points:], t=t_b)],
                    dim=-1)

                new_vals_b, pred_xstart = self._vb_terms_bpd(denoise_fn, data_start=x_start, data_t=data_t, t=t_b,
                                                             clip_denoised=clip_denoised, return_pred_xstart=True)

                # MSE for progressive prediction loss
                assert pred_xstart.shape == x_start[:, :, self.sv_points:].shape

                new_mse_b = ((pred_xstart - x_start[:, :, self.sv_points:]) ** 2).mean(
                    dim=list(range(1, len(pred_xstart.shape))))

                assert new_vals_b.shape == new_mse_b.shape == torch.Size([B])

                # Insert the calculated term into the tensor of all terms
                mask_bt = t_b[:, None] == torch.arange(T, device=t_b.device)[None, :].float()
                vals_bt_ = vals_bt_ * (~mask_bt) + new_vals_b[:, None] * mask_bt
                mse_bt_ = mse_bt_ * (~mask_bt) + new_mse_b[:, None] * mask_bt

                assert mask_bt.shape == vals_bt_.shape == vals_bt_.shape == torch.Size([B, T])

            prior_bpd_b = self._prior_bpd(x_start[:, :, self.sv_points:])
            total_bpd_b = vals_bt_.sum(dim=1) + prior_bpd_b

            assert vals_bt_.shape == mse_bt_.shape == torch.Size([B, T]) and \
                   total_bpd_b.shape == prior_bpd_b.shape == torch.Size([B])

            return total_bpd_b.mean(), vals_bt_.mean(), prior_bpd_b.mean(), mse_bt_.mean()


class PVCNN2(PVCNN2Base):
    num_n = 128 # Number of neighbors

    # Define set abstraction layers
    sa_blocks = [((32, 2, 32), (10240, 0.1, num_n, (32, 64))),
                 ((64, 3, 16), (2560, 0.2, num_n, (64, 128))),
                 ((128, 3, 8), (640, 0.4, num_n, (128, 256))),
                 (None, (160, 0.8, num_n, (256, 256, 512))),
                 ]

    # Define feature propagation layers
    fp_blocks = [((256, 256), (256, 3, 8)),
                 ((256, 256), (256, 3, 8)),
                 ((256, 128), (128, 2, 16)),
                 ((128, 128, 64), (64, 2, 32)),
                 ]

    def __init__(self, num_classes, sv_points, embed_dim, use_att, dropout, extra_feature_channels=3,
                 width_multiplier=1.0, voxel_resolution_multiplier=1.0):
        super().__init__(num_classes=num_classes, sv_points=sv_points, embed_dim=embed_dim, use_att=use_att,
                         dropout=dropout, extra_feature_channels=extra_feature_channels,
                         width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier)


class Model(nn.Module):
    def __init__(self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str,
                 width_mult: float, vox_res_mult: float, min_snr_gamma: float = 5.0):
        super(Model, self).__init__()

        # Create diffusion
        self.diffusion = GaussianDiffusion(betas, loss_type, model_mean_type, model_var_type,
                                           sv_points=(args.num_points - args.num_nn),
                                           min_snr_gamma=min_snr_gamma)

        # Create point-voxel-cnn network
        self.model = PVCNN2(num_classes=args.nc, sv_points=(args.num_points - args.num_nn), embed_dim=args.embed_dim,
                            use_att=args.attention, dropout=args.dropout, extra_feature_channels=0,
                            width_multiplier=width_mult, voxel_resolution_multiplier=vox_res_mult)

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(self._denoise, x0, clip_denoised)

        return {'total_bpd_b': total_bpd_b,
                'terms_bpd': vals_bt,
                'prior_bpd_b': prior_bpd_b,
                'mse_bt': mse_bt}

    def _denoise(self, data, t):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape

        # Sample random time t step for training
        t = torch.randint(0, self.diffusion.num_timesteps, size=(B,), device=data.device)

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        # Compute training loss
        losses = self.diffusion.p_losses(denoise_fn=self._denoise, data_start=data, t=t, noise=noises)

        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(self, partial_x, shape, device, noise_fn=torch.randn, clip_denoised=True, keep_running=False,
                    sampling_method='ddpm', sampling_steps=1000):
        """
        Generate samples using DDPM or DDIM.

        Args:
            partial_x: Partial point cloud (skull points)
            shape: Shape of noise to generate
            device: CUDA device
            noise_fn: Noise generation function
            clip_denoised: Whether to clip denoised values
            keep_running: Whether to run 2x num_timesteps (DDPM only)
            sampling_method: 'ddpm' or 'ddim'
            sampling_steps: Number of steps (DDIM only, DDPM always uses 1000)

        Returns:
            Generated samples with partial_x concatenated
        """
        if sampling_method == 'ddim':
            return self.diffusion.ddim_sample_loop(
                partial_x, self._denoise, shape=shape, device=device,
                noise_fn=noise_fn, clip_denoised=clip_denoised,
                sampling_steps=sampling_steps
            )
        elif sampling_method == 'ddpm':
            return self.diffusion.p_sample_loop(
                partial_x, self._denoise, shape=shape, device=device,
                noise_fn=noise_fn, clip_denoised=clip_denoised, keep_running=keep_running
            )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. Use 'ddpm' or 'ddim'.")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        betas = np.linspace(b_start, b_end, time_num)

    elif schedule_type == 'cosine':
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal 2021)
        steps = np.arange(time_num + 1, dtype=np.float64)
        alpha_bar = np.cos(((steps / time_num) + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = np.clip(betas, a_min=1e-6, a_max=0.999)

    elif schedule_type == 'warm0.1':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.2':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    elif schedule_type == 'warm0.5':
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)

    else:
        raise NotImplementedError(schedule_type)

    return betas


def get_dataset(num_points, num_nn, path, dataset, augment):
    if dataset == 'SkullBreak':
        tr_dataset = SkullBreakDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode='shape_bbox',
                                       augment=augment)
    else:
        tr_dataset = SkullFixDataset(path=path, num_points=num_points, num_nn=num_nn, norm_mode='shape_bbox',
                                     augment=augment)
    return tr_dataset


def get_dataloader(opt, train_dataset, test_dataset=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,  # Critical: ensures all ranks get same number of batches
        )
        test_sampler = (
            torch.utils.data.distributed.DistributedSampler(
                test_dataset, 
                num_replicas=world_size, 
                rank=rank,
                shuffle=False,
                drop_last=False,  # Keep all test samples
            )
            if test_dataset is not None else None
        )
    else:
        train_sampler = None
        test_sampler = None

    global_batch = opt.bs
    per_device_batch = max(global_batch // world_size, 1)
    num_workers = max(opt.workers // world_size, 0)
    persistent_workers = num_workers > 0
    prefetch_factor = opt.prefetch_factor if opt.prefetch_factor > 0 and num_workers > 0 else None

    train_loader_kwargs = dict(
        batch_size=per_device_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    if test_dataset is not None:
        test_loader_kwargs = dict(
            batch_size=per_device_batch,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=persistent_workers,
        )
        if prefetch_factor is not None:
            test_loader_kwargs['prefetch_factor'] = prefetch_factor

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            **test_loader_kwargs,
        )
    else:
        test_dataloader = None

    return train_dataloader, test_dataloader, train_sampler, test_sampler


def _get_git_commit_short() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _get_git_commit_full() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _compute_file_hash(filepath: str) -> Optional[str]:
    """Compute SHA256 hash of a file for reproducibility tracking."""
    try:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:16]  # First 16 chars for brevity
    except Exception:
        return None


def _get_gpu_info() -> Dict[str, Any]:
    """Gather GPU information for reproducibility."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
            })
    return info


def create_run_directory(base_dir: str, dataset: str, experiment_tag: Optional[str] = None) -> str:
    """
    Create a standardized run directory with the schema:

    runs/<dataset>/<timestamp>[-<tag>]/
        ├── checkpoints/      # Model checkpoints (best, latest, periodic)
        ├── logs/             # Training logs
        ├── metrics/          # Evaluation metrics (proxy, full)
        ├── samples/          # Generated samples for visualization
        └── run_metadata.json # Reproducibility metadata

    Returns the path to the created run directory.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{timestamp}"
    if experiment_tag:
        run_name = f"{timestamp}-{experiment_tag}"

    run_dir = os.path.join(base_dir, "runs", dataset, run_name)

    # Create subdirectories
    subdirs = ["checkpoints", "logs", "metrics", "samples"]
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

    return run_dir


def save_run_metadata(run_dir: str, opt: argparse.Namespace, extra_info: Optional[Dict] = None) -> str:
    """
    Save reproducibility metadata to run_metadata.json.

    Includes:
    - Git commit hash (full and short)
    - All CLI arguments
    - Dataset hash (CSV file hash)
    - GPU count and device info
    - Random seed
    - Timestamp
    - PyTorch and CUDA versions
    """
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": _get_git_commit_full(),
        "git_commit_short": _get_git_commit_short(),
        "seed": opt.manualSeed,
        "cli_args": vars(opt),
        "dataset": {
            "name": opt.dataset,
            "csv_path": opt.path,
            "csv_hash": _compute_file_hash(opt.path),
        },
        "gpu_info": _get_gpu_info(),
        "environment": {
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        },
        "run_directory": run_dir,
    }

    if extra_info:
        metadata.update(extra_info)

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path


def train(gpu, opt, output_dir, noises_init):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", gpu if gpu is not None else 0))
    should_diag = rank == 0

    # Use logs/ subdirectory for logging within the run directory
    log_dir = os.path.join(output_dir, "logs") if output_dir else None
    if should_diag and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger = setup_logging(log_dir)
        logger.info(f"Run directory: {output_dir}")
        logger.info(f"Checkpoints: {opt.checkpoint_dir}")
    else:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())

    cuda_available = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if opt.amp_dtype == 'bfloat16' else torch.float16
    if hasattr(opt, "disable_amp"):
        amp_enabled = not opt.disable_amp
    else:
        amp_enabled = getattr(opt, "amp", False)
    use_amp = cuda_available and amp_enabled
    use_grad_scaler = use_amp and amp_dtype == torch.float16

    # Use samples/ subdirectory for generated samples
    if should_diag:
        samples_dir = os.path.join(output_dir, "samples") if output_dir else None
        if samples_dir:
            os.makedirs(samples_dir, exist_ok=True)
            outf_syn = samples_dir
        else:
            outf_syn, = setup_output_subdirs(output_dir, 'syn')

    is_distributed = world_size > 1
    if is_distributed and cuda_available:
        torch.cuda.set_device(local_rank)
        
        # Initialize with a shorter timeout during development to fail faster
        # Default is 30 minutes; you can adjust via TORCH_DIST_TIMEOUT env var
        timeout_minutes = int(os.environ.get("TORCH_DIST_TIMEOUT", "30"))
        timeout = datetime.timedelta(minutes=timeout_minutes)
        
        dist.init_process_group(
            backend=opt.dist_backend,
            timeout=timeout
        )
        
        if should_diag:
            logger.info(f"Initialized distributed training: world_size={world_size}, "
                       f"rank={rank}, local_rank={local_rank}, timeout={timeout_minutes}min")

    ''' Dataset and data loader '''
    train_dataset = get_dataset(opt.num_points, opt.num_nn, opt.path, opt.dataset, opt.augment)
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

    # Log batch size information (critical for multi-GPU training verification)
    per_device_batch = dataloader.batch_size
    effective_global_batch = per_device_batch * world_size
    if should_diag:
        logger.info(f"Batch size: {per_device_batch} per GPU × {world_size} GPUs = {effective_global_batch} effective global batch")
        logger.info(f"Dataset: {len(train_dataset)} samples, {len(dataloader)} batches/epoch/rank")

    ''' Create networks '''
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type, opt.width_mult, opt.vox_res_mult,
                  min_snr_gamma=opt.min_snr_gamma)

    if cuda_available:
        model = model.cuda()

    use_compile = (not opt.disable_compile) and hasattr(torch, "compile")
    compile_applied = False
    if use_compile:
        compile_kwargs = {}
        if opt.compile_backend:
            compile_kwargs['backend'] = opt.compile_backend
        if opt.compile_mode:
            compile_kwargs['mode'] = opt.compile_mode
        if opt.compile_fullgraph:
            compile_kwargs['fullgraph'] = True

        try:
            model.model = torch.compile(model.model, **compile_kwargs)
            compile_applied = True
        except Exception as compile_err:
            compile_applied = False
            if should_diag:
                logger.warning(f"torch.compile failed ({compile_err}); falling back to eager execution")

    if is_distributed:
        def _transform_(m):
            return nn.parallel.DistributedDataParallel(
                m,
                device_ids=[local_rank] if cuda_available else None,
                output_device=local_rank if cuda_available else None,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
                static_graph=opt.ddp_static_graph,
            )

        model.multi_gpu_wrapper(_transform_)

    # Initialize EMA (Exponential Moving Average) for model parameters
    ema = None
    if opt.ema_decay > 0:
        ema = EMA(model, decay=opt.ema_decay)
        if should_diag:
            logger.info(f"EMA enabled with decay={opt.ema_decay}")

    if should_diag:
        logger.info(opt)
        if compile_applied:
            logger.info(f"torch.compile enabled for PVCNN backbone "
                        f"(backend={opt.compile_backend}, mode={opt.compile_mode or 'default'}, "
                        f"fullgraph={opt.compile_fullgraph})")
        elif use_compile:
            logger.info("torch.compile requested but running in eager mode")

    # Initialize wandb (only on main process)
    use_wandb = WANDB_AVAILABLE and not opt.no_wandb and should_diag
    if use_wandb:
        logger.info("W&B enabled. If you're not logged in yet, run `wandb login` (or set `WANDB_API_KEY`) before training.")
        wandb_name = opt.wandb_name or f"{opt.dataset}_ep{opt.niter}_bs{opt.bs}_lr{opt.lr}"
        try:
            wandb.init(
                project=opt.wandb_project,
                entity=opt.wandb_entity,
                name=wandb_name,
                config=vars(opt),
                resume='allow'
            )
            wandb.watch(model, log='all', log_freq=opt.print_freq * 10)
            logger.info(f"Wandb logging enabled: {wandb.run.url}")
        except Exception as wandb_err:
            use_wandb = False
            logger.warning(f"Wandb init failed; continuing without wandb. Error: {wandb_err}")

    base_batch = max(opt.lr_base_batch, 1)
    lr_scale = float(opt.bs) / float(base_batch)
    scaled_lr = opt.lr * lr_scale

    optimizer_kwargs = dict(
        lr=scaled_lr,
        weight_decay=opt.decay,
        betas=(opt.beta1, 0.999),
    )
    fused_enabled = False
    if cuda_available and not opt.no_fused_adam:
        optimizer_kwargs['fused'] = True

    if should_diag:
        logger.info(f"Effective LR scaled to {scaled_lr:.6e} using factor {lr_scale:.3f} "
                    f"(baseline batch={base_batch}, current batch={opt.bs})")

    try:
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
        fused_enabled = optimizer_kwargs.get('fused', False)
    except TypeError:
        optimizer_kwargs.pop('fused', None)
        optimizer = optim.Adam(model.parameters(), **optimizer_kwargs)
        fused_enabled = False

    warmup_epochs = max(opt.lr_warmup_epochs, 0)

    # Build the main LR scheduler (ExponentialLR or CosineAnnealingLR)
    if opt.lr_scheduler == 'cosine':
        cosine_T_max = max(opt.cosine_T_max, 1)
        cosine_eta_min = opt.cosine_eta_min
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_T_max, eta_min=cosine_eta_min)
        sched_desc = f"CosineAnnealingLR(T_max={cosine_T_max}, eta_min={cosine_eta_min:.1e})"
    else:
        main_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, opt.lr_gamma)
        sched_desc = f"ExponentialLR(gamma={opt.lr_gamma})"

    if warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=opt.lr_warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        lr_scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
        if should_diag:
            logger.info(f"Applying LR warmup for {warmup_epochs} epochs "
                        f"(start_factor={opt.lr_warmup_start_factor:.3f}) before {sched_desc}")
    else:
        lr_scheduler = main_scheduler
        if should_diag:
            logger.info(f"LR scheduler: {sched_desc} (no warmup)")
    scaler = GradScaler("cuda", enabled=use_grad_scaler)

    if should_diag:
        logger.info(f"AMP: {'on' if use_amp else 'off'} "
                    f"(dtype={opt.amp_dtype}, grad_scaler={'on' if use_grad_scaler else 'off'})")
        if cuda_available:
            if fused_enabled:
                logger.info("Using fused Adam optimizer kernels")
            elif not opt.no_fused_adam:
                logger.info("Fused Adam kernels unavailable; using standard Adam")

    # Initialize checkpoint manager
    ckpt_manager = None
    if should_diag:
        ckpt_manager = CheckpointManager(
            checkpoint_dir=opt.checkpoint_dir,
            keep_last_n=opt.keep_last_n,
            logger=logger
        )
        logger.info(f"Checkpoints will be saved to: {opt.checkpoint_dir}")
        logger.info(f"Checkpoint frequency: every {opt.checkpoint_freq} epochs, keeping last {opt.keep_last_n}")

    start_epoch = 0
    if opt.model:
        checkpoint = torch.load(opt.model, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        if start_epoch > 0 and should_diag:
            logger.info(f"Resuming from checkpoint {opt.model} at epoch {start_epoch}")
        # Load best loss from checkpoint for proper best model tracking
        if ckpt_manager:
            ckpt_manager.load_best_loss(opt.model)

    # Track best checkpoint path for W&B artifact upload at end (rank-0 only)
    best_ckpt_path_for_wandb = None
    best_ckpt_epoch_for_wandb = None
    best_ckpt_loss_for_wandb = None

    # Initialize gating loop tracker
    gating_config = GatingConfig(
        decision_epochs=opt.gating_decision_epochs_list,
        max_epochs=opt.gating_max_epochs,
        proxy_eval_freq=opt.gating_proxy_eval_freq,
        grad_norm_threshold=opt.gating_grad_norm_threshold,
        loss_spike_threshold=opt.gating_loss_spike_threshold,
        plateau_delta_threshold=opt.gating_plateau_delta,
        plateau_loss_variance_threshold=opt.gating_plateau_variance,
        enabled=opt.gating_enabled,
    )
    gating_tracker = GatingLoopTracker(gating_config, logger=logger if should_diag else None)

    if should_diag:
        logger.info(f"Gating loop: enabled={gating_config.enabled}, max_epochs={gating_config.max_epochs}, "
                    f"decision_epochs={gating_config.decision_epochs}")

    # Determine effective max epochs (respects both --niter and --gating-max-epochs)
    if gating_config.enabled:
        effective_max_epochs = min(opt.niter, gating_config.max_epochs)
        if should_diag:
            logger.info(f"Effective max epochs: {effective_max_epochs} (niter={opt.niter}, gating_max={gating_config.max_epochs})")
    else:
        effective_max_epochs = opt.niter

    # Track stop reason for final summary
    final_stop_reason = StopReason.MAX_EPOCHS
    final_stop_details = ""

    try:
        # Training loop
        for epoch in range(start_epoch, effective_max_epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            epoch_step_count = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            epoch_stats = EpochStats(epoch=epoch, lr=optimizer.param_groups[0]['lr'])
            gating_stop_triggered = False

            for i, data in enumerate(dataloader):
                pc_in = data['train_points'].transpose(1, 2)  # Input point cloud
                noises_batch = noises_init[data['idx']].transpose(1, 2)  # Noise (num_nn points)

                if cuda_available:
                    pc_in = pc_in.cuda(non_blocking=True)
                    noises_batch = noises_batch.cuda(non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    loss = model.get_loss_iter(pc_in, noises_batch).mean()

                loss_value = float(loss.detach().item())
                epoch_loss_sum += loss_value
                epoch_loss_count += 1
                epoch_stats.loss_values.append(loss_value)

                if use_grad_scaler:
                    scaler.scale(loss).backward()
                    if opt.grad_clip is not None:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    else:
                        scaler.unscale_(optimizer)
                        grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if opt.grad_clip is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
                    else:
                        grad_norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
                    optimizer.step()

                # Update EMA shadow parameters
                if ema is not None:
                    ema.update(model)

                grad_norm_value = float(grad_norm.item()) if torch.is_tensor(grad_norm) else float(grad_norm)
                epoch_stats.grad_norms.append(grad_norm_value)

                epoch_step_count += 1

                # Gating loop: per-step check for NaN/Inf/exploding gradients
                should_stop, stop_reason, stop_details = gating_tracker.step_check(loss_value, grad_norm_value)
                if should_stop:
                    if should_diag:
                        logger.error(f"GATING STOP at epoch {epoch}, step {i}: {stop_reason.value}")
                        logger.error(f"Details: {stop_details}")
                    final_stop_reason = stop_reason
                    final_stop_details = stop_details
                    gating_stop_triggered = True
                    break

                # Print progress
                if i % opt.print_freq == 0 and should_diag:
                    logger.info('[{:>3d}/{:>3d}][{:>3d}/{:>3d}]    loss: {:>10.4f},    grad_norm: {:>10.4f}'
                                .format(epoch, effective_max_epochs, i, len(dataloader), loss_value, grad_norm_value))

                    # Log to wandb
                    if use_wandb:
                        wandb.log({
                            "train/loss": loss_value,
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch,
                            "train/lr": optimizer.param_groups[0]['lr']
                        }, step=epoch * len(dataloader) + i)

            # If gating stop was triggered during the epoch, break out of training loop
            if gating_stop_triggered:
                break
            
            # Validate that all ranks completed the same number of steps
            if is_distributed:
                step_device = torch.device('cuda', local_rank) if cuda_available else torch.device('cpu')
                step_count_tensor = torch.tensor([epoch_step_count], dtype=torch.long, device=step_device)
                gathered_counts = [torch.zeros_like(step_count_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_counts, step_count_tensor)
                
                # Check for divergence on rank 0
                if should_diag:
                    counts = [t.item() for t in gathered_counts]
                    if len(set(counts)) > 1:
                        logger.error(f"STEP COUNT DIVERGENCE at epoch {epoch}! Counts per rank: {counts}")
                        logger.error("This indicates uneven data distribution. Check your dataset and sampler.")
                        raise RuntimeError(f"Step count mismatch across ranks: {counts}")
                    else:
                        if epoch % 100 == 0:  # Log periodically
                            logger.info(f"Epoch {epoch}: All ranks completed {epoch_step_count} steps ✓")
            
            lr_scheduler.step()

            # Evaluate
            if (epoch + 1) % opt.diagIter == 0:
                # CRITICAL: All ranks must participate to avoid NCCL hangs
                if is_distributed:
                    dist.barrier()  # Synchronize before diagnostics
                
                if should_diag:
                    logger.info('Diagnosis:')

                    x_range = [pc_in.min().item(), pc_in.max().item()]
                    kl_stats = model.all_kl(pc_in)
                    logger.info('      [{:>3d}/{:>3d}]    '
                                'x_range: [{:>10.4f}, {:>10.4f}],   '
                                'total_bpd_b: {:>10.4f},    '
                                'terms_bpd: {:>10.4f},  '
                                'prior_bpd_b: {:>10.4f}    '
                                'mse_bt: {:>10.4f}  '
                                .format(epoch, opt.niter,
                                        *x_range,
                                        kl_stats['total_bpd_b'].item(),
                                        kl_stats['terms_bpd'].item(), kl_stats['prior_bpd_b'].item(),
                                        kl_stats['mse_bt'].item()))
                    
                    # Log diagnostics to wandb
                    if use_wandb:
                        wandb.log({
                            "diag/total_bpd": kl_stats['total_bpd_b'].item(),
                            "diag/terms_bpd": kl_stats['terms_bpd'].item(),
                            "diag/prior_bpd": kl_stats['prior_bpd_b'].item(),
                            "diag/mse": kl_stats['mse_bt'].item(),
                            "diag/x_min": x_range[0],
                            "diag/x_max": x_range[1]
                        }, step=epoch * len(dataloader))
                
                if is_distributed:
                    dist.barrier()  # Synchronize after diagnostics

            # Visualize some samples
            if (epoch + 1) % opt.vizIter == 0:
                # CRITICAL: All ranks must participate to avoid NCCL hangs
                if is_distributed:
                    dist.barrier()  # Synchronize before visualization
                
                if should_diag:
                    logger.info('Generation: eval')

                    model.eval()

                    with torch.no_grad():
                        x_gen_eval = model.gen_samples(pc_in[:, :, :(opt.num_points - opt.num_nn)],
                                                       pc_in[:, :, (opt.num_points - opt.num_nn):].shape,
                                                       pc_in.device, clip_denoised=False).detach().cpu()

                        gen_stats = [x_gen_eval.mean(), x_gen_eval.std()]
                        gen_eval_range = [x_gen_eval.min().item(), x_gen_eval.max().item()]

                        logger.info('      [{:>3d}/{:>3d}]  '
                                    'eval_gen_range: [{:>10.4f}, {:>10.4f}]     '
                                    'eval_gen_stats: [mean={:>10.4f}, std={:>10.4f}]      '
                                    .format(epoch, opt.niter, *gen_eval_range, *gen_stats))
                        
                        # Log generation stats to wandb
                        if use_wandb:
                            wandb.log({
                                "gen/mean": gen_stats[0].item(),
                                "gen/std": gen_stats[1].item(),
                                "gen/min": gen_eval_range[0],
                                "gen/max": gen_eval_range[1]
                            }, step=epoch * len(dataloader))

                    # Save samples and ground truth
                    export_to_pc_batch('%s/epoch_%03d_samples_eval' % (outf_syn, epoch),
                                       (x_gen_eval.transpose(1, 2)).numpy())

                    export_to_pc_batch('%s/epoch_%03d_ground_truth' % (outf_syn, epoch),
                                       (pc_in.transpose(1, 2).detach().cpu()).numpy())

                    export_to_pc_batch('%s/epoch_%03d_partial' % (outf_syn, epoch),
                                       (pc_in[:, :, :(opt.num_points - opt.num_nn)].transpose(1, 2).detach().cpu()).numpy())

                    model.train()
                
                if is_distributed:
                    dist.barrier()  # Synchronize after visualization

            # Compute epoch average loss
            epoch_avg_loss = epoch_loss_sum / max(epoch_loss_count, 1)

            # Record epoch statistics for gating loop
            gating_tracker.record_epoch(epoch_stats)

            # Determine if this is a decision epoch (save checkpoint regardless of periodic setting)
            is_decision_epoch = gating_tracker.is_decision_epoch(epoch)

            # Save checkpoints using checkpoint manager (use EMA weights if available)
            if should_diag and ckpt_manager:
                # Save at decision epochs (50, 100, 200, 500, 700) AND periodic intervals
                is_periodic = ((epoch + 1) % opt.checkpoint_freq == 0) or is_decision_epoch
                if ema is not None:
                    ema.apply_shadow(model)
                save_info = ckpt_manager.save(model, optimizer, epoch, epoch_avg_loss, is_periodic=is_periodic)
                if ema is not None:
                    ema.restore(model)
                if save_info.get("saved_best"):
                    best_ckpt_path_for_wandb = save_info.get("best_path")
                    best_ckpt_epoch_for_wandb = epoch
                    best_ckpt_loss_for_wandb = epoch_avg_loss

                if is_decision_epoch:
                    logger.info(f"Decision checkpoint saved at epoch {epoch}")

            # Log epoch summary to wandb
            if use_wandb and should_diag:
                loss_summary = gating_tracker.get_loss_summary(50)
                wandb.log({
                    "epoch/loss_mean": epoch_stats.loss_mean,
                    "epoch/loss_median": epoch_stats.loss_median,
                    "epoch/grad_norm_mean": epoch_stats.grad_norm_mean,
                    "epoch/grad_norm_max": epoch_stats.grad_norm_max,
                    "gating/loss_median_50ep": loss_summary.get("median", 0),
                    "gating/loss_p90_50ep": loss_summary.get("p90", 0),
                    "gating/loss_p10_50ep": loss_summary.get("p10", 0),
                }, step=epoch * len(dataloader))

            # Proxy evaluation every proxy_eval_freq epochs (rank-0 only)
            is_proxy_eval_epoch = gating_tracker.is_proxy_eval_epoch(epoch)
            if is_proxy_eval_epoch and opt.proxy_eval_enabled:
                # Barrier before proxy eval - ALL ranks must participate
                if is_distributed:
                    dist.barrier()

                # Only rank 0 runs the actual proxy evaluation
                if should_diag:
                    if PROXY_EVAL_AVAILABLE:
                        logger.info(f"=== PROXY EVALUATION at epoch {epoch} ===")
                        proxy_device = torch.device(f'cuda:{local_rank}') if cuda_available else torch.device('cpu')

                        # Run proxy evaluation
                        proxy_metrics = run_proxy_evaluation(
                            pcdiff_model=model,
                            vox_config_path=Path(opt.proxy_eval_vox_config),
                            vox_checkpoint_path=Path(opt.proxy_eval_vox_checkpoint),
                            subset_path=Path(opt.proxy_eval_subset),
                            device=proxy_device,
                            num_points=opt.num_points,
                            num_nn=opt.num_nn,
                            num_ens=opt.proxy_eval_num_ens,
                            sampling_method=opt.proxy_eval_sampling_method,
                            sampling_steps=opt.proxy_eval_sampling_steps,
                            base_dir=Path.cwd(),
                            logger=logger,
                        )

                        # Record proxy metrics in gating tracker
                        if "error" not in proxy_metrics:
                            gating_tracker.record_proxy_metrics(ProxyMetrics(
                                epoch=epoch,
                                dsc=proxy_metrics["dsc"],
                                bdsc=proxy_metrics["bdsc"],
                                hd95=proxy_metrics["hd95"],
                            ))

                            # Save proxy metrics to file
                            if output_dir:
                                save_proxy_metrics(proxy_metrics, epoch, Path(output_dir), logger)

                            # Log to wandb
                            if use_wandb:
                                wandb.log({
                                    "proxy/dsc": proxy_metrics["dsc"],
                                    "proxy/bdsc": proxy_metrics["bdsc"],
                                    "proxy/hd95": proxy_metrics["hd95"],
                                    "proxy/epoch": epoch,
                                }, step=epoch * len(dataloader))

                            logger.info(f"  DSC={proxy_metrics['dsc']:.4f}, bDSC={proxy_metrics['bdsc']:.4f}, "
                                        f"HD95={proxy_metrics['hd95']:.2f}")
                        else:
                            logger.warning(f"  Proxy eval failed: {proxy_metrics.get('error', 'unknown')}")

                        logger.info("=" * 50)

                        # Ensure model is back in training mode
                        model.train()
                    else:
                        logger.warning("Proxy evaluation skipped: voxelization dependencies not available")

                # Barrier after proxy eval - ALL ranks must participate
                if is_distributed:
                    dist.barrier()

            # Gating decision at decision epochs
            if is_decision_epoch and gating_config.enabled:
                stop_reason, stop_details = gating_tracker.evaluate_gating_decision(epoch)

                if should_diag:
                    loss_summary = gating_tracker.get_loss_summary(50)
                    logger.info(f"=== GATING DECISION at epoch {epoch} ===")
                    logger.info(f"  Loss (last 50 ep): median={loss_summary['median']:.4f}, "
                                f"p10={loss_summary['p10']:.4f}, p90={loss_summary['p90']:.4f}")
                    logger.info(f"  Decision: {stop_reason.value}")
                    logger.info(f"  Details: {stop_details}")
                    logger.info("=" * 50)

                    # Log gating decision to wandb
                    if use_wandb:
                        wandb.log({
                            "gating/decision_epoch": epoch,
                            "gating/decision": stop_reason.value,
                        }, step=epoch * len(dataloader))

                if stop_reason != StopReason.CONTINUE:
                    final_stop_reason = stop_reason
                    final_stop_details = stop_details

                    # Save gating state before breaking
                    if should_diag and output_dir:
                        gating_state_path = os.path.join(output_dir, "metrics", "gating_state.json")
                        gating_tracker.save_state(gating_state_path)
                        logger.info(f"Gating state saved to: {gating_state_path}")

                    break

            if is_distributed:
                dist.barrier()

        # Training loop completed (either normally or via gating stop)
        if should_diag:
            training_summary = gating_tracker.get_training_summary()
            logger.info(f"=== TRAINING COMPLETED ===")
            logger.info(f"  Final epoch: {training_summary['last_epoch']}")
            logger.info(f"  Stop reason: {final_stop_reason.value}")
            logger.info(f"  Details: {final_stop_details}")
            logger.info(f"  Loss summary (last 50 ep): {training_summary['loss_summary_last_50']}")

            # Save final gating state
            if output_dir:
                gating_state_path = os.path.join(output_dir, "metrics", "gating_state.json")
                gating_tracker.save_state(gating_state_path)
                logger.info(f"Final gating state saved to: {gating_state_path}")

            # Log final summary to wandb
            if use_wandb:
                wandb.log({
                    "final/stop_reason": final_stop_reason.value,
                    "final/total_epochs": training_summary['last_epoch'] + 1,
                    "final/loss_median": training_summary['loss_summary_last_50'].get('median', 0),
                })

    finally:
        # Upload best checkpoint as a W&B artifact (rank-0 only), so it can be downloaded later.
        if use_wandb and should_diag and best_ckpt_path_for_wandb and os.path.exists(best_ckpt_path_for_wandb):
            try:
                commit = _get_git_commit_short()
                artifact_name = f"pcdiff-{opt.dataset}-best"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="model",
                    metadata={
                        "dataset": opt.dataset,
                        "epoch": best_ckpt_epoch_for_wandb,
                        "loss": best_ckpt_loss_for_wandb,
                        "checkpoint_file": os.path.basename(best_ckpt_path_for_wandb),
                        "git_commit": commit,
                    },
                )
                artifact.add_file(best_ckpt_path_for_wandb)
                wandb.log_artifact(artifact, aliases=["best", f"epoch_{best_ckpt_epoch_for_wandb}"])
                logger.info(f"Logged W&B artifact '{artifact_name}' (aliases: best, epoch_{best_ckpt_epoch_for_wandb})")
            except Exception as artifact_err:
                logger.warning(f"Failed to log W&B artifact for best checkpoint: {artifact_err}")

        # Finish wandb run
        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass

        if is_distributed:
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def main():
    opt = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    # Set CUDA device early for all ranks
    if torch.cuda.is_available() and is_distributed:
        torch.cuda.set_device(local_rank)

    # Create standardized run directory (rank-0 only creates, others wait)
    if rank == 0:
        # Use pcdiff/ as base directory for runs
        base_dir = os.path.dirname(__file__)
        run_dir = create_run_directory(base_dir, opt.dataset, opt.experiment_tag)

        # Update checkpoint directory to be inside run directory
        opt.checkpoint_dir = os.path.join(run_dir, "checkpoints")

        # Save reproducibility metadata
        save_run_metadata(run_dir, opt)

        # Also create the legacy output_dir for backward compatibility (samples)
        output_dir = run_dir

        # Copy source file to run directory for reference
        copy_source(__file__, run_dir)

        print(f"[Rank 0] Created run directory: {run_dir}")
    else:
        # Non-rank-0 processes: use a placeholder that will be synced
        output_dir = None
        run_dir = None

    # Synchronize run directory across all ranks using a file-based approach
    # to avoid double process group initialization
    if is_distributed:
        # Use a simple file-based sync for run_dir instead of creating process group twice
        # Rank 0 writes the path to a temp file, other ranks wait and read it
        import tempfile
        import time

        # Use a predictable temp file path based on master port
        master_port = os.environ.get("MASTER_PORT", "29500")
        sync_file = f"/tmp/pcdiff_run_dir_sync_{master_port}.txt"

        if rank == 0:
            # Write run_dir to sync file
            with open(sync_file, 'w') as f:
                f.write(run_dir)
        else:
            # Wait for rank 0 to write the file
            max_wait = 60  # seconds
            waited = 0
            while not os.path.exists(sync_file) and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1

            if os.path.exists(sync_file):
                with open(sync_file, 'r') as f:
                    run_dir = f.read().strip()
                output_dir = run_dir
                opt.checkpoint_dir = os.path.join(run_dir, "checkpoints")
            else:
                raise RuntimeError(f"Rank {rank}: Timed out waiting for run_dir sync file")

    ''' Initialization '''
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(opt.matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = not opt.disable_tf32
        torch.backends.cudnn.allow_tf32 = not opt.disable_tf32
    seed_everything(opt.manualSeed, deterministic=not opt.nondeterministic)
    noises_init = torch.randn(570, opt.num_nn, 3)  # Init noise (num_nn random points)

    train(opt.gpu, opt, output_dir, noises_init)

    # Cleanup sync file (rank 0 only)
    if is_distributed and rank == 0:
        master_port = os.environ.get("MASTER_PORT", "29500")
        sync_file = f"/tmp/pcdiff_run_dir_sync_{master_port}.txt"
        try:
            os.remove(sync_file)
        except OSError:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="set the path to the dataset here")
    parser.add_argument('--dataset', type=str, required=True, help="specify the used dataset (SkullBreak or SkullFix)")

    # Data loader parameters
    parser.add_argument('--bs', type=int, default=8, help='input batch size')
    parser.add_argument('--workers', type=int, default=24, help='workers dataloader')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='prefetch batches per DataLoader worker (0 disables prefetching)')
    parser.add_argument('--niter', type=int, default=15000, help='number of epochs to train for')

    # Input point cloud
    parser.add_argument('--nc', type=int, default=3, help="dimension of one point (usually 3 for x, y,z)")
    parser.add_argument('--num_points', type=int, default=30720, help="number of points the point cloud should contain")
    parser.add_argument('--num_nn', type=int, default=3072, help="number of points that represent the implant")

    ''' Model '''
    # Diffusion process parameters (variance schedule, number of steps)
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--schedule_type', type=str, default='linear')
    parser.add_argument('--time_num', type=int, default=1000, help='number of timesteps T in diffusion process')
    parser.add_argument('--augment', type=eval, default=False, help='apply random rotation (+-10deg) around all axes')

    # Model parameters
    parser.add_argument('--attention', type=eval, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mse_minsnr', 'kl'],
                        help="loss type: 'mse', 'mse_minsnr' (Min-SNR-γ weighted), or 'kl'")
    parser.add_argument('--min_snr_gamma', type=float, default=5.0,
                        help='Min-SNR-γ clipping value for mse_minsnr loss (paper default: 5.0)')
    parser.add_argument('--model_mean_type', type=str, default='eps')
    parser.add_argument('--model_var_type', type=str, default='fixedsmall')
    parser.add_argument('--vox_res_mult', type=float, default=1.0)
    parser.add_argument('--width_mult', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate for E, default=0.0002')
    parser.add_argument('--lr-base-batch', type=int, default=8,
                        help='baseline global batch size that opt.lr was tuned for (used for linear scaling)')
    parser.add_argument('--lr-warmup-epochs', type=int, default=500,
                        help='number of epochs for linear LR warmup (0 disables warmup)')
    parser.add_argument('--lr-warmup-start-factor', type=float, default=0.01,
                        help='initial LR factor relative to scaled LR during warmup')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=None,
                        help='max gradient norm (None disables clipping)')
    parser.add_argument('--ema_decay', type=float, default=0.0,
                        help='EMA decay rate (0 disables EMA, 0.9999 recommended for diffusion)')
    parser.add_argument('--lr_gamma', type=float, default=1, help='lr decay for ExponentialLR scheduler')
    parser.add_argument('--lr_scheduler', type=str, default='exponential', choices=['exponential', 'cosine'],
                        help="LR scheduler type: 'exponential' (ExponentialLR) or 'cosine' (CosineAnnealingLR)")
    parser.add_argument('--cosine_T_max', type=int, default=700,
                        help='T_max for CosineAnnealingLR (number of epochs for one cosine cycle)')
    parser.add_argument('--cosine_eta_min', type=float, default=1e-6,
                        help='minimum LR for CosineAnnealingLR')
    parser.add_argument('--disable-compile', action='store_true',
                        help='disable torch.compile even if it is available')
    parser.add_argument('--compile-backend', type=str, default='inductor',
                        help='backend to use for torch.compile')
    parser.add_argument('--compile-mode', type=str, default=None,
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='preset for torch.compile() optimizations')
    parser.add_argument('--compile-fullgraph', action='store_true',
                        help='request fullgraph compilation for torch.compile')
    parser.add_argument('--amp', dest='amp', action='store_true',
                        help='enable automatic mixed precision (requires AMP-compatible CUDA ops)')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                        help='disable AMP (default, recommended unless CUDA ops support low precision)')
    parser.add_argument('--disable-amp', dest='amp', action='store_false',
                        help=argparse.SUPPRESS)
    parser.add_argument('--amp-dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                        help='dtype to use for autocast when AMP is enabled')
    parser.set_defaults(amp=False)
    parser.add_argument('--no-fused-adam', action='store_true',
                        help='disable fused Adam optimizer kernel')
    parser.add_argument('--ddp-static-graph', action='store_true',
                        help='enable static_graph optimization for DistributedDataParallel')
    parser.add_argument('--nondeterministic', action='store_true',
                        help='allow faster but non-deterministic CUDA algorithms')
    parser.add_argument('--disable-tf32', action='store_true',
                        help='disable TF32 matmul/tensor core usage')
    parser.add_argument('--matmul-precision', type=str, default='high', choices=['high', 'medium', 'low'],
                        help='torch.set_float32_matmul_precision value')

    # Model path (for continuing the training of existing models)
    parser.add_argument('--model', default='', help="path to model (to continue training)")

    # Checkpoint management
    parser.add_argument('--checkpoint_dir', type=str, default='pcdiff/checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--checkpoint_freq', type=int, default=5,
                        help='save checkpoint every N epochs')
    parser.add_argument('--keep_last_n', type=int, default=3,
                        help='number of periodic checkpoints to keep (excludes best model)')

    # Distributed training (torchrun handles world size / rank via environment variables)
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use (auto-detected from LOCAL_RANK if not specified)')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='torch.distributed backend to use when launched via torchrun')

    ''' Evaluation '''
    parser.add_argument('--saveIter', type=int, default=1000, help='unit: epoch')
    parser.add_argument('--diagIter', type=int, default=2000, help='unit: epoch')
    parser.add_argument('--vizIter', type=int, default=2000, help='unit: epoch')
    parser.add_argument('--print_freq', type=int, default=10, help='unit: iter')

    # Manual seed for deterministic sampling, etc.
    parser.add_argument('--manualSeed', default=1234, type=int, help='random seed')

    # Experiment tracking with Weights & Biases
    parser.add_argument('--wandb-project', type=str, default='pcdiff-implant', help='wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='wandb entity/team name')
    parser.add_argument('--wandb-name', type=str, default=None, help='wandb run name (auto-generated if not set)')
    parser.add_argument('--no-wandb', action='store_true', help='disable wandb logging even if installed')

    # Run directory and experiment tagging
    parser.add_argument('--experiment-tag', type=str, default=None,
                        help='optional tag appended to run directory name (e.g., "paper-parity", "sqrt-lr")')

    # Gating loop configuration (700-epoch gating with decision checkpoints)
    parser.add_argument('--gating-enabled', type=eval, default=True,
                        help='enable the 700-epoch gating loop with early stopping')
    parser.add_argument('--gating-max-epochs', type=int, default=700,
                        help='maximum epochs for gating loop (hard cap)')
    parser.add_argument('--gating-decision-epochs', type=str, default='50,100,200,500,700',
                        help='comma-separated list of decision checkpoint epochs')
    parser.add_argument('--gating-proxy-eval-freq', type=int, default=50,
                        help='run proxy evaluation every N epochs')
    parser.add_argument('--gating-grad-norm-threshold', type=float, default=1e6,
                        help='gradient norm threshold for exploding gradient detection')
    parser.add_argument('--gating-loss-spike-threshold', type=float, default=10.0,
                        help='loss spike threshold as multiple of running median')
    parser.add_argument('--gating-plateau-delta', type=float, default=0.005,
                        help='minimum proxy metric improvement to avoid plateau detection')
    parser.add_argument('--gating-plateau-variance', type=float, default=0.3,
                        help='loss variance threshold (90p-10p)/median for plateau detection')

    # Proxy evaluation arguments
    parser.add_argument('--proxy-eval-enabled', type=eval, default=True,
                        help='enable proxy evaluation every N epochs (default: True)')
    parser.add_argument('--proxy-eval-subset', type=str,
                        default='pcdiff/proxy_validation_subset.json',
                        help='path to proxy validation subset JSON file')
    parser.add_argument('--proxy-eval-vox-config', type=str,
                        default='voxelization/configs/gen_skullbreak.yaml',
                        help='path to voxelization config YAML')
    parser.add_argument('--proxy-eval-vox-checkpoint', type=str,
                        default='voxelization/checkpoints/model_best.pt',
                        help='path to voxelization model checkpoint')
    parser.add_argument('--proxy-eval-num-ens', type=int, default=1,
                        help='ensemble size for proxy eval (1 for speed, 5 for full eval)')
    parser.add_argument('--proxy-eval-sampling-method', type=str, default='ddim',
                        choices=['ddim', 'ddpm'],
                        help='sampling method for proxy eval (ddim for speed)')
    parser.add_argument('--proxy-eval-sampling-steps', type=int, default=50,
                        help='sampling steps for proxy eval (50 for DDIM, 1000 for DDPM)')

    # Parse arguments
    opt = parser.parse_args()

    # Parse gating decision epochs from comma-separated string
    opt.gating_decision_epochs_list = [int(x.strip()) for x in opt.gating_decision_epochs.split(',')]

    return opt


if __name__ == '__main__':
    main()
