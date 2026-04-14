#!/usr/bin/env python3
"""
DIM-88: Wodzinski et al. 2024 Baseline - Volumetric Residual UNet
for Cranial Defect Reconstruction on SkullBreak.

Reference: "Improving Deep Learning-based Automatic Cranial Defect
Reconstruction by Heavy Data Augmentation" (arXiv 2406.06372)

Architecture: Volumetric Residual UNet (no augmentation baseline)
Input: Defective skull volume (256^3 binary)
Output: Predicted implant volume (256^3 binary)
Loss: Soft Dice Loss (same as paper)
Metric: Dice Score Coefficient (DSC)
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
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom, binary_dilation, binary_erosion, rotate
from sklearn.model_selection import train_test_split

BASE_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/datasets/SkullBreak")
RESULTS_DIR = Path("/mnt/data/home/mamuke588/pcdiff-implant/wodzinski_baseline")

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
    """Volumetric Residual UNet - Wodzinski et al. 2024 baseline."""
    def __init__(self, in_channels=1, base_filters=32):
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


def dice_score(pred, target, smooth=1.0):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ======================= Dataset =======================

class SkullBreakDataset(Dataset):
    def __init__(self, case_ids, base_dir, defect_type="bilateral",
                 resolution=256, augment=False):
        self.base_dir = Path(base_dir)
        self.defect_type = defect_type
        self.resolution = resolution
        self.augment = augment
        self.cases = []

        complete_dir = self.base_dir / "complete_skull"
        defect_dir = self.base_dir / "defective_skull" / defect_type

        for cid in case_ids:
            complete_path = complete_dir / f"{cid}.nrrd"
            defect_files = list(defect_dir.glob(f"{cid}_*.nrrd"))
            if not defect_files:
                defect_path = defect_dir / f"{cid}.nrrd"
            else:
                defect_path = defect_files[0]

            if complete_path.exists() and defect_path.exists():
                self.cases.append((str(complete_path), str(defect_path)))

        print(f"Dataset: {len(self.cases)} cases from {defect_type}")

    def __len__(self):
        return len(self.cases)

    def _resample(self, vol, target_shape):
        if vol.shape == target_shape:
            return vol
        factors = [t / s for t, s in zip(target_shape, vol.shape)]
        return zoom(vol, factors, order=1).astype(np.float32)

    def _augment(self, defective, implant):
        # Random flip along sagittal axis
        if random.random() > 0.5:
            defective = defective[::-1, :, :]
            implant = implant[::-1, :, :]
        # Random rotation in axial plane (small angles)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            defective = rotate(defective, angle, axes=(0, 1), reshape=False, order=1)
            implant = rotate(implant, angle, axes=(0, 1), reshape=False, order=1)
        return defective, implant

    def __getitem__(self, idx):
        complete_path, defect_path = self.cases[idx]

        complete, _ = nrrd.read(complete_path)
        defective, _ = nrrd.read(defect_path)

        # Binary
        complete = (complete > 0).astype(np.float32)
        defective = (defective > 0).astype(np.float32)

        # Implant = complete - defective
        implant = np.clip(complete - defective, 0, 1)

        # Resample to target resolution
        shape = (self.resolution,) * 3
        defective = self._resample(defective, shape)
        implant = self._resample(implant, shape)

        # Binarize after resampling
        defective = (defective > 0.5).astype(np.float32)
        implant = (implant > 0.5).astype(np.float32)

        if self.augment:
            defective, implant = self._augment(defective, implant)
            defective = (defective > 0.5).astype(np.float32)
            implant = (implant > 0.5).astype(np.float32)

        # Add channel dim
        defective = defective[np.newaxis]
        implant = implant[np.newaxis]

        return torch.from_numpy(defective), torch.from_numpy(implant)


def get_case_ids(base_dir, defect_type):
    """Get matching case IDs from complete and defective dirs."""
    complete_dir = Path(base_dir) / "complete_skull"
    defect_dir = Path(base_dir) / "defective_skull" / defect_type

    complete_ids = set()
    for f in complete_dir.glob("*.nrrd"):
        complete_ids.add(f.stem)

    defect_ids = set()
    for f in defect_dir.glob("*.nrrd"):
        # Extract base case ID (before first _ if present, or filename)
        stem = f.stem
        base_id = stem.split("_")[0] if "_" in stem else stem
        defect_ids.add(base_id)

    common = sorted(complete_ids & defect_ids)
    print(f"Complete: {len(complete_ids)}, Defective({defect_type}): {len(defect_ids)}, Common: {len(common)}")
    return common


# ======================= Training =======================

def train(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(RESULTS_DIR / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Get case IDs
    case_ids = get_case_ids(args.base_dir, args.defect_type)
    if len(case_ids) < 5:
        print(f"ERROR: Only {len(case_ids)} cases found. Need at least 5.")
        return

    # Train/val/test split (80/10/10)
    train_ids, test_ids = train_test_split(case_ids, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    # Save splits
    with open(RESULTS_DIR / "splits.json", "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f)

    # Datasets
    train_ds = SkullBreakDataset(train_ids, args.base_dir, args.defect_type,
                                  args.resolution, augment=True)
    val_ds = SkullBreakDataset(val_ids, args.base_dir, args.defect_type,
                                args.resolution, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet3D(in_channels=1, base_filters=args.base_filters).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = SoftDiceLoss()

    best_val_dice = 0.0
    log_lines = []

    for epoch in range(1, args.epochs + 1):
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

            if (batch_idx + 1) % 10 == 0:
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

            avg_val_loss = val_loss / val_batches
            avg_val_dice = val_dice_sum / val_batches

            lr_now = optimizer.param_groups[0]["lr"]
            log_line = (f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} "
                       f"train_dice={avg_train_dice:.4f} "
                       f"val_loss={avg_val_loss:.4f} val_dice={avg_val_dice:.4f} "
                       f"lr={lr_now:.6f}")
            print(log_line)
            log_lines.append(log_line)

            if avg_val_dice > best_val_dice:
                best_val_dice = avg_val_dice
                torch.save(model.state_dict(), RESULTS_DIR / "model_best.pt")
                print(f"  *NEW BEST* val_dice={best_val_dice:.4f}")

            # Save checkpoint
            if epoch % args.checkpoint_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                }, RESULTS_DIR / f"checkpoint_epoch{epoch}.pt")

    # Save final model
    torch.save(model.state_dict(), RESULTS_DIR / "model_final.pt")

    # Write log
    with open(RESULTS_DIR / "log.txt", "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nTraining complete. Best val DSC: {best_val_dice:.4f}")
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=str(BASE_DIR))
    parser.add_argument("--defect_type", default="bilateral")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--checkpoint_every", type=int, default=25)
    args = parser.parse_args()
    train(args)
