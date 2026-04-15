"""
Rectified Flow for Cranial Implant Generation

Based on:
- Zhou et al. 2025 "Rectified Flow for Efficient Automatic Implant Generation" (CGI 2024)
- Liu et al. 2023 "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"

Architecture: Same PVCNN backbone as pcdiff, but with flow-matching training objective.
"""

import argparse
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys; sys.path.insert(0, "pcdiff"); from train_completion import PVCNN2
from modules import Swish  # noqa
from datasets.skullbreak_data import SkullBreakDataset
from datasets.skullfix_data import SkullFixDataset


class RectifiedFlow:
    def __init__(self, sv_points, sigma_min=1e-5):
        self.sv_points = sv_points
        self.sigma_min = sigma_min

    def sample_t(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def interpolate(self, x_0, x_1, t):
        t = t.view(-1, 1, 1)
        return (1 - t) * x_0 + t * x_1

    def target_velocity(self, x_0, x_1):
        return x_1 - x_0

    def training_losses(self, model, data_start, t=None):
        B, D, N = data_start.shape

        if t is None:
            t = self.sample_t(B, data_start.device)

        skull = data_start[:, :, :self.sv_points]
        implant = data_start[:, :, self.sv_points:]

        x_0 = torch.randn_like(implant)
        x_1 = implant

        x_t = self.interpolate(x_0, x_1, t)
        v_target = self.target_velocity(x_0, x_1)

        model_input = torch.cat([skull, x_t], dim=-1)
        t_discrete = (t * 999).long().clamp(0, 999)

        model_output = model(model_input, t_discrete)
        v_pred = model_output[:, :, self.sv_points:]

        losses = ((v_pred - v_target) ** 2).mean(dim=list(range(1, len(data_start.shape))))
        return losses

    @torch.no_grad()
    def euler_sample(self, model, partial_x, shape, device, num_steps=1):
        x = torch.randn(shape, dtype=torch.float, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            t_discrete = (t * 999).long().clamp(0, 999)

            model_input = torch.cat([partial_x, x], dim=-1)
            v_pred = model(model_input, t_discrete)[:, :, self.sv_points:]
            x = x + v_pred * dt

        return torch.cat([partial_x, x], dim=-1)


def get_dataset(data_path, dataset_name, num_points, num_nn, augment=False):
    if dataset_name == 'SkullBreak':
        return SkullBreakDataset(path=data_path, num_points=num_points, num_nn=num_nn,
                                 norm_mode='shape_bbox', augment=augment)
    elif dataset_name == 'SkullFix':
        return SkullFixDataset(path=data_path, num_points=num_points, num_nn=num_nn,
                               norm_mode='shape_bbox', augment=augment)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train():
    parser = argparse.ArgumentParser(description='Rectified Flow Training for Cranial Implant')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='SkullBreak', choices=['SkullBreak', 'SkullFix'])
    parser.add_argument('--save-dir', type=str, default='runs/rectified_flow')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--ema-decay', type=float, default=0.9999)
    parser.add_argument('--num-points', type=int, default=30720)
    parser.add_argument('--num-nn', type=int, default=3072)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--width-mult', type=float, default=1.0)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=50)
    parser.add_argument('--sample-interval', type=int, default=200)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.gpus > 1:
        dist.init_process_group('nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    sv_points = args.num_points - args.num_nn

    dataset = get_dataset(args.data_dir, args.dataset, args.num_points, args.num_nn, args.augment)

    if args.gpus > 1:
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                               num_workers=4, pin_memory=True, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)

    model = PVCNN2(use_att=True, dropout=0.1, extra_feature_channels=0,
        num_classes=3, embed_dim=args.embed_dim,
        width_multiplier=args.width_mult, voxel_resolution_multiplier=1,
        sv_points=sv_points,
    ).to(device)

    ema_model = PVCNN2(use_att=True, dropout=0.1, extra_feature_channels=0,
        num_classes=3, embed_dim=args.embed_dim,
        width_multiplier=args.width_mult, voxel_resolution_multiplier=1,
        sv_points=sv_points,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    if args.gpus > 1:
        model = DDP(model, device_ids=[local_rank])

    rf = RectifiedFlow(sv_points=sv_points)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if args.gpus > 1 else model
        raw_model.load_state_dict(ckpt['state_dict'])
        ema_model.load_state_dict(ckpt['ema_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('loss', float('inf'))
        if local_rank == 0:
            print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if local_rank == 0:
        print(f"Training RF: {len(dataset)} samples, batch_size={args.batch_size}, "
              f"gpus={args.gpus}, num_points={args.num_points}")

    for epoch in range(start_epoch, args.epochs):
        if args.gpus > 1:
            sampler.set_epoch(epoch)

        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            combined = batch["train_points"].to(device).permute(0, 2, 1)

            losses = rf.training_losses(model, combined)
            loss = losses.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                for p_ema, p in zip(ema_model.parameters(),
                                    model.module.parameters() if args.gpus > 1 else model.parameters()):
                    p_ema.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

            epoch_losses.append(loss.item())

            if (batch_idx + 1) % args.log_interval == 0 and local_rank == 0:
                print(f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] loss={loss.item():.6f}")

        scheduler.step()
        epoch_time = time.time() - t0

        if local_rank == 0:
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch} avg_loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.2e} "
                  f"time={epoch_time:.1f}s")

            if (epoch + 1) % args.save_interval == 0 or avg_loss < best_loss:
                ckpt = {
                    'epoch': epoch,
                    'state_dict': (model.module if args.gpus > 1 else model).state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'loss': avg_loss,
                    'args': vars(args),
                }

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(ckpt, os.path.join(args.save_dir, 'model_best.pt'))
                    print(f"  New best loss: {best_loss:.6f}")

                if (epoch + 1) % args.save_interval == 0:
                    torch.save(ckpt, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))

            if (epoch + 1) % args.sample_interval == 0:
                ema_model.eval()
                sample_batch = combined[:1]
                skull = sample_batch[:, :, :sv_points]
                shape = (1, 3, args.num_nn)

                result_1step = rf.euler_sample(ema_model, skull, shape, device, num_steps=1)
                result_5step = rf.euler_sample(ema_model, skull, shape, device, num_steps=5)

                np.save(os.path.join(args.save_dir, f'sample_1step_epoch{epoch}.npy'),
                       result_1step.cpu().numpy())
                np.save(os.path.join(args.save_dir, f'sample_5step_epoch{epoch}.npy'),
                       result_5step.cpu().numpy())
                print(f"  Saved sample outputs (1-step and 5-step)")

    if args.gpus > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    train()
