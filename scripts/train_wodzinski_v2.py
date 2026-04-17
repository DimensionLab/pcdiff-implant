#!/usr/bin/env python3
"""
DIM-88: Wodzinski v2 - Improved Volumetric Residual UNet
for Cranial Defect Reconstruction on SkullBreak.

Key improvements over v1 baseline:
1. Train on ALL 5 defect types (not just bilateral)
2. Heavy geometric augmentation (flips, affine, crops, binary noise)
   matching Wodzinski et al. 2024 paper
3. Multi-GPU training with DDP
4. Combined SoftDice + BCE loss
5. LR warmup + cosine schedule

Reference: "Improving Deep Learning-based Automatic Cranial Defect
Reconstruction by Heavy Data Augmentation" (arXiv 2406.06372)
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

import numpy as np
import nrrd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from scipy.ndimage import zoom, rotate, affine_transform
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/datasets/SkullBreak")
RESULTS_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/wodzinski_v2")

DEFECT_TYPES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]


# ======================= Model =======================

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm3d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = ResidualBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        s = self.block(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.block = ResidualBlock3D(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatch
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        dd = skip.shape[4] - x.shape[4]
        x = F.pad(x, [dd//2, dd-dh//2, dw//2, dw-dw//2, dh//2, dh-dh//2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualUNet3D(nn.Module):
    """Volumetric Residual UNet - Wodzinski et al. 2024."""
    def __init__(self, in_channels=2, base_filters=32):
        super().__init__()
        # Encoder
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_filters * 8, base_filters * 16)

        # Decoder
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters)

        # Output
        self.out_conv = nn.Conv3d(base_filters, 1, 1)

    def forward(self, x):
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


# ======================= Loss =======================

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined SoftDice + BCE loss for better convergence."""
    def __init__(self, dice_weight=0.7, bce_weight=0.3):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target)


def dice_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ======================= Augmentation =======================

class HeavyGeometricAugmentation:
    """
    Heavy geometric augmentation matching Wodzinski et al. 2024.
    Applied with random probability 0.75 each, in random order:
    1. Random flips (all axes)
    2. Random affine (rotation, translation, scaling)
    3. Random crops
    4. Random binary noise
    """
    def __init__(self, prob=0.75, resolution=256):
        self.prob = prob
        self.resolution = resolution

    def __call__(self, defective, implant):
        # Random flips along each axis
        for axis in range(3):
            if random.random() < self.prob:
                defective = np.flip(defective, axis=axis).copy()
                implant = np.flip(implant, axis=axis).copy()

        # Random rotation in axial plane (larger range than v1)
        if random.random() < self.prob:
            angle = random.uniform(-30, 30)
            defective = rotate(defective, angle, axes=(0, 1), reshape=False, order=1)
            implant = rotate(implant, angle, axes=(0, 1), reshape=False, order=1)

        # Random rotation in sagittal plane
        if random.random() < self.prob:
            angle = random.uniform(-20, 20)
            defective = rotate(defective, angle, axes=(1, 2), reshape=False, order=1)
            implant = rotate(implant, angle, axes=(1, 2), reshape=False, order=1)

        # Random scaling (0.85-1.15)
        if random.random() < self.prob:
            scale = random.uniform(0.85, 1.15)
            defective = zoom(defective, scale, order=1)
            implant = zoom(implant, scale, order=1)
            # Crop/pad back to target size
            defective = self._crop_or_pad(defective, self.resolution)
            implant = self._crop_or_pad(implant, self.resolution)

        # Random binary noise (add/remove small patches)
        if random.random() < self.prob:
            noise_prob = random.uniform(0.001, 0.01)
            noise = np.random.random(defective.shape) < noise_prob
            defective = np.clip(defective + noise.astype(np.float32), 0, 1)

        return defective, implant

    def _crop_or_pad(self, vol, target_size):
        """Crop or pad volume to target size."""
        result = np.zeros((target_size, target_size, target_size), dtype=vol.dtype)
        slices_src = []
        slices_dst = []
        for i in range(3):
            if vol.shape[i] >= target_size:
                start = (vol.shape[i] - target_size) // 2
                slices_src.append(slice(start, start + target_size))
                slices_dst.append(slice(0, target_size))
            else:
                start = (target_size - vol.shape[i]) // 2
                slices_src.append(slice(0, vol.shape[i]))
                slices_dst.append(slice(start, start + vol.shape[i]))
        result[tuple(slices_dst)] = vol[tuple(slices_src)]
        return result


# ======================= Dataset =======================

class SkullBreakMultiDefectDataset(Dataset):
    """
    Dataset loading from ALL defect types.
    Optionally includes defect type as a second input channel.
    """
    def __init__(self, case_ids_with_defects, base_dir, resolution=256,
                 augment=False, defect_type_channel=True):
        self.base_dir = Path(base_dir)
        self.resolution = resolution
        self.augment = augment
        self.defect_type_channel = defect_type_channel
        self.augmenter = HeavyGeometricAugmentation(prob=0.75, resolution=resolution) if augment else None
        self.cases = []

        complete_dir = self.base_dir / "complete_skull"

        for case_id, defect_type in case_ids_with_defects:
            complete_path = complete_dir / f"{case_id}.nrrd"
            defect_dir = self.base_dir / "defective_skull" / defect_type
            implant_dir = self.base_dir / "implant" / defect_type

            defect_files = list(defect_dir.glob(f"{case_id}*.nrrd"))
            implant_files = list(implant_dir.glob(f"{case_id}*.nrrd"))

            if not defect_files:
                defect_path = defect_dir / f"{case_id}.nrrd"
            else:
                defect_path = defect_files[0]

            if not implant_files:
                implant_path = implant_dir / f"{case_id}.nrrd"
            else:
                implant_path = implant_files[0]

            if complete_path.exists() and defect_path.exists() and implant_path.exists():
                self.cases.append((str(complete_path), str(defect_path), str(implant_path), defect_type))

        # Create defect type mapping
        self.defect_map = {dt: i for i, dt in enumerate(DEFECT_TYPES)}

        print(f"Dataset: {len(self.cases)} cases from {len(DEFECT_TYPES)} defect types")

    def __len__(self):
        return len(self.cases)

    def _resample(self, vol, target_shape):
        if vol.shape == target_shape:
            return vol
        factors = [t / s for t, s in zip(target_shape, vol.shape)]
        return zoom(vol, factors, order=1).astype(np.float32)

    def __getitem__(self, idx):
        complete_path, defect_path, implant_path, defect_type = self.cases[idx]

        complete, _ = nrrd.read(complete_path)
        defective, _ = nrrd.read(defect_path)
        implant, _ = nrrd.read(implant_path)

        # Binary
        complete = (complete > 0).astype(np.float32)
        defective = (defective > 0).astype(np.float32)
        implant = (implant > 0).astype(np.float32)

        # Resample to target resolution
        shape = (self.resolution,) * 3
        defective = self._resample(defective, shape)
        implant = self._resample(implant, shape)

        # Binarize after resampling
        defective = (defective > 0.5).astype(np.float32)
        implant = (implant > 0.5).astype(np.float32)

        # Apply augmentation
        if self.augmenter is not None:
            defective, implant = self.augmenter(defective, implant)
            defective = (defective > 0.5).astype(np.float32)
            implant = (implant > 0.5).astype(np.float32)

        # Stack defective skull + defect type mask as input channels
        if self.defect_type_channel:
            dt_idx = self.defect_map[defect_type]
            dt_mask = np.full_like(defective, dt_idx / (len(DEFECT_TYPES) - 1))
            inp = np.stack([defective, dt_mask], axis=0)
        else:
            inp = defective[np.newaxis]

        return torch.from_numpy(inp), torch.from_numpy(implant[np.newaxis])


def get_all_case_ids(base_dir):
    """Get all case IDs with all defect types."""
    base_dir = Path(base_dir)
    complete_dir = base_dir / "complete_skull"

    complete_ids = set()
    for f in complete_dir.glob("*.nrrd"):
        complete_ids.add(f.stem)

    cases = []
    for defect_type in DEFECT_TYPES:
        defect_dir = base_dir / "defective_skull" / defect_type
        defect_ids = set()
        for f in defect_dir.glob("*.nrrd"):
            stem = f.stem
            base_id = stem.split("_")[0] if "_" in stem else stem
            defect_ids.add(base_id)

        common = sorted(complete_ids & defect_ids)
        for cid in common:
            cases.append((cid, defect_type))

    print(f"Total cases: {len(cases)} ({len(complete_ids)} skulls x {len(DEFECT_TYPES)} defect types)")
    return cases


# ======================= Training =======================

def setup_ddp():
    """Setup DDP environment."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
    return rank, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    rank, world_size = setup_ddp()
    is_main = (rank == 0)

    if is_main:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # Get all case IDs with defect types
    all_cases = get_all_case_ids(args.base_dir)
    if len(all_cases) < 10:
        print(f"ERROR: Only {len(all_cases)} cases found. Need at least 10.")
        return

    # Split by case ID (not by defect type) to ensure no data leakage
    unique_ids = sorted(set(cid for cid, _ in all_cases))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    train_cases = [(cid, dt) for cid, dt in all_cases if cid in train_set]
    val_cases = [(cid, dt) for cid, dt in all_cases if cid in val_set]
    test_cases = [(cid, dt) for cid, dt in all_cases if cid in test_set]

    if is_main:
        print(f"Train: {len(train_cases)} cases, Val: {len(val_cases)} cases, Test: {len(test_cases)} cases")
        with open(RESULTS_DIR / "splits.json", "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

    # Datasets
    train_ds = SkullBreakMultiDefectDataset(train_cases, args.base_dir, args.resolution, augment=True)
    val_ds = SkullBreakMultiDefectDataset(val_cases, args.base_dir, args.resolution, augment=False)

    # Data loaders with DDP sampler
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               shuffle=(train_sampler is None), num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device(f"cuda:{rank}")
    in_channels = 2 if args.defect_type_channel else 1
    model = ResidualUNet3D(in_channels=in_channels, base_filters=args.base_filters).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params:,}")
        print(f"Input channels: {in_channels} (defect_type_channel={args.defect_type_channel})")

    # Optimizer with LR warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # LR scheduler: warmup + cosine
    warmup_epochs = min(20, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3)

    best_val_dice = 0.0
    log_lines = []

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_dice_sum = 0.0
        n_batches = 0

        for batch_idx, (defective, implant) in enumerate(train_loader):
            defective = defective.to(device)
            implant = implant.to(device)

            optimizer.zero_grad()
            pred = model(defective)
            loss = criterion(pred, implant)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dice_sum += dice_score(pred.detach(), implant)
            n_batches += 1

            if is_main and (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / n_batches
        avg_train_dice = train_dice_sum / n_batches

        # Validation
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0.0
            val_dice_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for defective, implant in val_loader:
                    defective = defective.to(device)
                    implant = implant.to(device)
                    pred = model(defective)
                    loss = criterion(pred, implant)
                    val_loss += loss.item()
                    val_dice_sum += dice_score(pred, implant)
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            avg_val_dice = val_dice_sum / max(val_batches, 1)

            # Gather metrics from all GPUs
            if world_size > 1:
                metrics = torch.tensor([avg_val_dice, avg_val_loss, val_batches],
                                       device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                avg_val_dice = metrics[0].item() / world_size
                avg_val_loss = metrics[1].item() / world_size

            lr_now = optimizer.param_groups[0]["lr"]
            log_line = (f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
                       f"train_dice={avg_train_dice:.4f} "
                       f"val_loss={avg_val_loss:.4f} val_dice={avg_val_dice:.4f} "
                       f"lr={lr_now:.6f}")

            if is_main:
                print(log_line)
                log_lines.append(log_line)

                if avg_val_dice > best_val_dice:
                    best_val_dice = avg_val_dice
                    raw_model = model.module if world_size > 1 else model
                    torch.save(raw_model.state_dict(), RESULTS_DIR / "model_best.pt")
                    print(f"  *NEW BEST* val_dice={best_val_dice:.4f}")

                # Save checkpoint periodically
                if epoch % args.checkpoint_every == 0:
                    raw_model = model.module if world_size > 1 else model
                    torch.save({
                        "epoch": epoch,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_val_dice": best_val_dice,
                    }, RESULTS_DIR / f"checkpoint_epoch{epoch}.pt")

    # Save final model
    if is_main:
        raw_model = model.module if world_size > 1 else model
        torch.save(raw_model.state_dict(), RESULTS_DIR / "model_final.pt")

        with open(RESULTS_DIR / "log.txt", "w") as f:
            f.write("\n".join(log_lines))

        print(f"\nTraining complete. Best val DSC: {best_val_dice:.4f}")
        print(f"Results saved to {RESULTS_DIR}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(BASE_DIR))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--defect_type_channel", action="store_true", default=True)
    parser.add_argument("--no_defect_type_channel", dest="defect_type_channel", action="store_false")
    args = parser.parse_args()
    train(args)
