#!/usr/bin/env python3
"""
Wodzinski et al. 2024 — Volumetric Residual UNet for Cranial Defect Reconstruction.

Reference: "Improving Deep Learning-based Automatic Cranial Defect Reconstruction
by Heavy Data Augmentation" (arXiv:2406.06372)

Architecture: Volumetric Residual UNet (no augmentation baseline)
Input: Defective skull volume (256^3 binary)
Output: Predicted implant volume (256^3 binary, soft probabilities)

Internally called "cran-2" in the CrAInial application.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================= Model Architecture =======================


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
        # Handle size mismatch from pooling
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        dd = skip.shape[4] - x.shape[4]
        x = F.pad(x, [dd // 2, dd - dd // 2, dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualUNet3D(nn.Module):
    """Volumetric Residual UNet — Wodzinski et al. 2024 baseline.

    Args:
        in_channels: Input channels (1 for binary volume).
        base_filters: Base number of filters (32 = 26.8M params, as published).
    """

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
        self.output_conv = nn.Conv3d(base_filters, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.sigmoid(self.output_conv(d1))


# ======================= Inference =======================


def load_wodzinski_model(checkpoint_path, device, base_filters=32):
    """Load a trained Wodzinski ResidualUNet3D from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: torch.device to load model on.
        base_filters: Must match training config (32 for published baseline).

    Returns:
        model: ResidualUNet3D in eval mode on device.
    """
    model = ResidualUNet3D(in_channels=1, base_filters=base_filters)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()
    return model


def preprocess_volume(volume, target_resolution=256):
    """Preprocess a volume for Wodzinski model input.

    Args:
        volume: numpy array (H, W, D) — defective skull binary volume.
        target_resolution: Target resolution (default 256).

    Returns:
        tensor: (1, 1, R, R, R) float tensor on CPU.
    """
    from scipy.ndimage import zoom

    # Binarize
    vol = (volume > 0).astype(np.float32)

    # Resize if needed
    if vol.shape != (target_resolution, target_resolution, target_resolution):
        factors = [target_resolution / s for s in vol.shape]
        vol = zoom(vol, factors, order=1)

    # To tensor (1, 1, R, R, R)
    tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()
    return tensor


def postprocess_output(output_tensor, threshold=0.5):
    """Postprocess model output to binary implant volume.

    Args:
        output_tensor: (1, 1, R, R, R) soft prediction.
        threshold: Binarization threshold (default 0.5).

    Returns:
        volume: numpy array (R, R, R) uint8 binary volume.
    """
    pred = output_tensor.squeeze().cpu().numpy()
    return (pred > threshold).astype(np.uint8)


def run_wodzinski_inference(model, volume, device, threshold=0.5):
    """Run Wodzinski inference on a defective skull volume.

    Args:
        model: Loaded ResidualUNet3D model in eval mode.
        volume: numpy array (H, W, D) — defective skull binary volume.
        device: torch.device.
        threshold: Binarization threshold.

    Returns:
        implant_volume: numpy uint8 array (256, 256, 256) — predicted implant.
        inference_time: float — inference time in seconds.
    """
    import time

    input_tensor = preprocess_volume(volume).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start

    implant_volume = postprocess_output(output, threshold=threshold)
    return implant_volume, inference_time


def volume_to_mesh(volume, spacing=(1.0, 1.0, 1.0)):
    """Convert binary volume to mesh using marching cubes.

    Args:
        volume: numpy uint8 array (R, R, R).
        spacing: Voxel spacing in mm.

    Returns:
        vertices: numpy array (N, 3) in world coordinates.
        faces: numpy array (M, 3) triangle indices.
    """
    from skimage.measure import marching_cubes

    vertices, faces, normals, _ = marching_cubes(volume, level=0.5, spacing=spacing)
    return vertices, faces
