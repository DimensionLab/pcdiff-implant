#!/usr/bin/env python3
"""
DIM-88: Wodzinski v3 — Improved cranial reconstruction with:
1. Extreme geometric augmentation (matching paper Table 1 "Extreme" settings)
2. Symmetry loss (from Wodzinski symmetry enforcement paper, 2411.17342)
3. Boundary Dice loss for print-ready surface quality
4. Deeper architecture (base_filters=48)
5. Multi-defect training on all 5 SkullBreak defect types
6. Multi-GPU DDP training

References:
- arXiv 2406.06372 (heavy augmentation)
- arXiv 2411.17342 (symmetry enforcement)
"""

import os
import sys
import json
import time
import math
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
from scipy.ndimage import zoom, rotate, binary_dilation, binary_erosion
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/datasets/SkullBreak")
RESULTS_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/wodzinski_v3")

DEFECT_TYPES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]

# Defect type encoding for the second input channel
DEFECT_TYPE_MAP = {dt: i for i, dt in enumerate(DEFECT_TYPES)}


# ======================= Model =======================

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
        ) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + self.shortcut(x))


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ResidualBlock3D(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        s = self.block(x)
        p = self.pool(s)
        return s, p


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.block = ResidualBlock3D(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=2, base_filters=48, dropout=0.1):
        super().__init__()
        bf = base_filters
        self.enc1 = EncoderBlock(in_channels, bf)
        self.enc2 = EncoderBlock(bf, bf * 2, dropout)
        self.enc3 = EncoderBlock(bf * 2, bf * 4, dropout)
        self.enc4 = EncoderBlock(bf * 4, bf * 8, dropout)
        self.bottleneck = ResidualBlock3D(bf * 8, bf * 16, dropout)
        self.dec4 = DecoderBlock(bf * 16, bf * 8, bf * 8, dropout)
        self.dec3 = DecoderBlock(bf * 8, bf * 4, bf * 4, dropout)
        self.dec2 = DecoderBlock(bf * 4, bf * 2, bf * 2)
        self.dec1 = DecoderBlock(bf * 2, bf, bf)
        self.out_conv = nn.Conv3d(bf, 1, 1)

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


# ======================= Losses =======================

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


class BoundaryDiceLoss(nn.Module):
    """Surface/boundary dice loss for print-ready surface quality."""
    def __init__(self, smooth=1.0, thickness=3):
        super().__init__()
        self.smooth = smooth
        self.thickness = thickness

    def forward(self, pred, target):
        pred_bin = (torch.sigmoid(pred) > 0.5).float()
        pred_boundary = self._extract_boundary(pred_bin, self.thickness)
        target_boundary = self._extract_boundary(target, self.thickness)
        pred_soft = torch.sigmoid(pred)
        pred_boundary_soft = pred_soft * pred_boundary
        intersection = (pred_boundary_soft * target_boundary).sum(dim=(2, 3, 4))
        union = pred_boundary_soft.sum(dim=(2, 3, 4)) + target_boundary.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    @staticmethod
    def _extract_boundary(vol, thickness):
        kernel_size = 2 * thickness + 1
        padding = thickness
        pool = F.max_pool3d(vol, kernel_size, stride=1, padding=padding)
        eroded = -F.max_pool3d(-vol, kernel_size, stride=1, padding=padding)
        return pool - eroded


class SymmetryLoss(nn.Module):
    """
    Symmetry enforcement loss (Wodzinski 2411.17342).
    Reflects the reconstructed skull about the sagittal plane and computes
    DSC between the original and reflected reconstruction.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, defective):
        pred_prob = torch.sigmoid(pred)
        reconstructed = torch.clamp(defective + pred_prob, 0, 1)
        reflected = torch.flip(reconstructed, dims=[4])
        intersection = (reconstructed * reflected).sum(dim=(2, 3, 4))
        union = reconstructed.sum(dim=(2, 3, 4)) + reflected.sum(dim=(2, 3, 4))
        sym_dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - sym_dice.mean()


class CombinedLossV3(nn.Module):
    def __init__(self, dice_w=0.5, bce_w=0.2, boundary_w=0.15, symmetry_w=0.15):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryDiceLoss()
        self.symmetry = SymmetryLoss()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.boundary_w = boundary_w
        self.symmetry_w = symmetry_w

    def forward(self, pred, target, defective_skull=None):
        loss = self.dice_w * self.dice(pred, target) + self.bce_w * self.bce(pred, target)
        if self.boundary_w > 0:
            loss = loss + self.boundary_w * self.boundary(pred, target)
        if self.symmetry_w > 0 and defective_skull is not None:
            loss = loss + self.symmetry_w * self.symmetry(pred, defective_skull)
        return loss


def dice_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ======================= Augmentation =======================

class ExtremeGeometricAugmentation:
    """
    Extreme augmentation matching Wodzinski 2024 Table 1 "Extreme":
    - Flips: Sagittal, Frontal, Longitudinal
    - Crops: yes
    - Affine: 45 degrees, 15 voxels translation, 0.4-1.3 scale
    - Binary noise: 1.8-4.5 structuring element, 1.0 threshold
    """
    def __init__(self, prob=0.75, resolution=256):
        self.prob = prob
        self.resolution = resolution

    def __call__(self, defective, implant):
        ops = [self._flip, self._affine, self._crop, self._binary_noise]
        random.shuffle(ops)
        for op in ops:
            if random.random() < self.prob:
                defective, implant = op(defective, implant)
        return defective, implant

    def _flip(self, defective, implant):
        for axis in range(3):
            if random.random() < 0.5:
                defective = np.flip(defective, axis=axis).copy()
                implant = np.flip(implant, axis=axis).copy()
        return defective, implant

    def _affine(self, defective, implant):
        # Rotation up to 45 degrees in random plane
        angle = random.uniform(-45, 45)
        axes_pairs = [(0, 1), (0, 2), (1, 2)]
        axes = random.choice(axes_pairs)
        defective = rotate(defective, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)
        implant = rotate(implant, angle, axes=axes, reshape=False, order=1, mode='constant', cval=0)

        # Translation up to 15 voxels
        shift = [random.uniform(-15, 15) for _ in range(3)]
        from scipy.ndimage import shift as nd_shift
        defective = nd_shift(defective, shift, order=1, mode='constant', cval=0)
        implant = nd_shift(implant, shift, order=1, mode='constant', cval=0)

        # Scale 0.4-1.3
        scale = random.uniform(0.4, 1.3)
        if abs(scale - 1.0) > 0.02:
            defective = zoom(defective, scale, order=1, mode='constant', cval=0)
            implant = zoom(implant, scale, order=1, mode='constant', cval=0)
            defective = self._crop_or_pad(defective, self.resolution)
            implant = self._crop_or_pad(implant, self.resolution)

        return defective, implant

    def _crop(self, defective, implant):
        crop_frac = random.uniform(0.05, 0.2)
        crop_vox = int(self.resolution * crop_frac)
        axis = random.randint(0, 2)
        side = random.choice(['start', 'end'])
        slices = [slice(None)] * 3
        if side == 'start':
            slices[axis] = slice(crop_vox, None)
        else:
            slices[axis] = slice(None, -crop_vox)
        defective = defective[tuple(slices)]
        implant = implant[tuple(slices)]
        defective = self._crop_or_pad(defective, self.resolution)
        implant = self._crop_or_pad(implant, self.resolution)
        return defective, implant

    def _binary_noise(self, defective, implant):
        # Morphological binary noise with structuring element 1.8-4.5
        se_size = random.uniform(1.8, 4.5)
        se_radius = max(1, int(se_size / 2))
        if random.random() < 0.5:
            defective = binary_dilation(defective > 0.5, iterations=se_radius).astype(np.float32)
        else:
            defective = binary_erosion(defective > 0.5, iterations=se_radius).astype(np.float32)
        return defective, implant

    def _crop_or_pad(self, vol, target_size):
        result = np.zeros((target_size,) * 3, dtype=vol.dtype)
        slices_src, slices_dst = [], []
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
    def __init__(self, case_ids_with_defects, base_dir, resolution=256,
                 augment=False, defect_type_channel=True):
        self.base_dir = Path(base_dir)
        self.resolution = resolution
        self.augment = augment
        self.defect_type_channel = defect_type_channel
        self.cases = case_ids_with_defects
        self.augmentor = ExtremeGeometricAugmentation(resolution=resolution) if augment else None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id, defect_type = self.cases[idx]
        complete_path = self.base_dir / "complete_skull" / f"{case_id}.nrrd"
        defect_path = self.base_dir / "defective_skull" / defect_type / f"{case_id}.nrrd"

        complete, _ = nrrd.read(str(complete_path))
        defective, _ = nrrd.read(str(defect_path))

        complete = (complete > 0).astype(np.float32)
        defective = (defective > 0).astype(np.float32)
        implant = np.clip(complete - defective, 0, 1)

        shape = (self.resolution,) * 3
        if defective.shape != shape:
            defective = self._resample(defective, shape)
            implant = self._resample(implant, shape)

        defective = (defective > 0.5).astype(np.float32)
        implant = (implant > 0.5).astype(np.float32)

        if self.augmentor is not None:
            defective, implant = self.augmentor(defective, implant)
            defective = (defective > 0.5).astype(np.float32)
            implant = (implant > 0.5).astype(np.float32)

        channels = [defective[np.newaxis]]
        if self.defect_type_channel:
            dt_channel = np.full_like(defective, DEFECT_TYPE_MAP[defect_type] / (len(DEFECT_TYPES) - 1))
            channels.append(dt_channel[np.newaxis])

        input_tensor = np.concatenate(channels, axis=0)
        return torch.from_numpy(input_tensor), torch.from_numpy(implant[np.newaxis])

    @staticmethod
    def _resample(vol, target_shape):
        factors = tuple(t / s for t, s in zip(target_shape, vol.shape))
        return zoom(vol, factors, order=1, mode='constant', cval=0)


def get_all_case_ids(base_dir):
    base_dir = Path(base_dir)
    complete_dir = base_dir / "complete_skull"
    complete_ids = {f.stem for f in complete_dir.glob("*.nrrd")}

    all_cases = []
    for dt in DEFECT_TYPES:
        defect_dir = base_dir / "defective_skull" / dt
        if not defect_dir.exists():
            continue
        for f in defect_dir.glob("*.nrrd"):
            base_id = f.stem.split("_")[0] if "_" in f.stem else f.stem
            if base_id in complete_ids:
                all_cases.append((base_id, dt))
    return all_cases


# ======================= DDP =======================

def setup_ddp():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ======================= Training =======================

def train(args):
    rank, world_size, local_rank = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    results_dir = Path(args.results_dir)
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    all_cases = get_all_case_ids(args.base_dir)
    if is_main:
        print(f"Total cases across all defect types: {len(all_cases)}")

    unique_ids = sorted(set(cid for cid, _ in all_cases))
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42)

    train_set, val_set = set(train_ids), set(val_ids)
    train_cases = [(cid, dt) for cid, dt in all_cases if cid in train_set]
    val_cases = [(cid, dt) for cid, dt in all_cases if cid in val_set]

    if is_main:
        print(f"Train: {len(train_cases)}, Val: {len(val_cases)}, Test: {len(test_ids)}")
        with open(results_dir / "splits.json", "w") as f:
            json.dump({"train": train_ids, "val": val_ids, "test": list(set(unique_ids) - train_set - val_set)}, f)

    train_ds = SkullBreakMultiDefectDataset(train_cases, args.base_dir, args.resolution,
                                             augment=True, defect_type_channel=args.defect_type_channel)
    val_ds = SkullBreakMultiDefectDataset(val_cases, args.base_dir, args.resolution,
                                           augment=False, defect_type_channel=args.defect_type_channel)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                               shuffle=(train_sampler is None), num_workers=args.num_workers,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    in_channels = 2 if args.defect_type_channel else 1
    model = ResidualUNet3D(in_channels=in_channels, base_filters=args.base_filters,
                            dropout=args.dropout).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model params: {n_params:,}, base_filters={args.base_filters}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    warmup_epochs = min(20, args.epochs // 10)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = CombinedLossV3(
        dice_w=args.dice_w, bce_w=args.bce_w,
        boundary_w=args.boundary_w, symmetry_w=args.symmetry_w
    )

    best_val_dice = 0.0
    log_lines = []
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0.0
        train_dice_sum = 0.0
        n_batches = 0

        for batch_idx, (inputs, implant) in enumerate(train_loader):
            inputs = inputs.to(device)
            implant = implant.to(device)

            optimizer.zero_grad()
            pred = model(inputs)

            defective_skull = inputs[:, 0:1] if args.symmetry_w > 0 else None
            loss = criterion(pred, implant, defective_skull)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_dice_sum += dice_score(pred.detach(), implant)
            n_batches += 1

            if is_main and (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] loss={loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_dice = train_dice_sum / max(n_batches, 1)

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            val_loss = 0.0
            val_dice_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for inputs, implant in val_loader:
                    inputs = inputs.to(device)
                    implant = implant.to(device)
                    pred = model(inputs)
                    defective_skull = inputs[:, 0:1] if args.symmetry_w > 0 else None
                    loss = criterion(pred, implant, defective_skull)
                    val_loss += loss.item()
                    val_dice_sum += dice_score(pred, implant)
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            avg_val_dice = val_dice_sum / max(val_batches, 1)

            if world_size > 1:
                metrics = torch.tensor([avg_val_dice, avg_val_loss, float(val_batches)], device=device)
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
                    patience_counter = 0
                    raw_model = model.module if world_size > 1 else model
                    torch.save(raw_model.state_dict(), results_dir / "model_best.pt")
                    print(f"  *NEW BEST* val_dice={best_val_dice:.4f}")
                else:
                    patience_counter += 1

                if epoch % args.checkpoint_every == 0:
                    raw_model = model.module if world_size > 1 else model
                    torch.save({
                        "epoch": epoch,
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_val_dice": best_val_dice,
                    }, results_dir / f"checkpoint_epoch{epoch}.pt")

                with open(results_dir / "log.txt", "w") as f:
                    f.write("\n".join(log_lines))

    if is_main:
        raw_model = model.module if world_size > 1 else model
        torch.save(raw_model.state_dict(), results_dir / "model_final.pt")
        print(f"\nTraining complete. Best val DSC: {best_val_dice:.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=str(RESULTS_DIR))
    parser.add_argument("--base_dir", default=str(BASE_DIR))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=50)
    parser.add_argument("--defect_type_channel", action="store_true", default=True)
    parser.add_argument("--no_defect_type_channel", dest="defect_type_channel", action="store_false")
    parser.add_argument("--dice_w", type=float, default=0.5)
    parser.add_argument("--bce_w", type=float, default=0.2)
    parser.add_argument("--boundary_w", type=float, default=0.15)
    parser.add_argument("--symmetry_w", type=float, default=0.15)
    args = parser.parse_args()
    train(args)
