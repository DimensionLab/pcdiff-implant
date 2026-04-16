#!/usr/bin/env python3
"""
Wodzinski et al. 2024 — Volumetric Residual UNet for Cranial Defect Reconstruction.

Reference: "Improving Deep Learning-based Automatic Cranial Defect Reconstruction
by Heavy Data Augmentation" (arXiv:2406.06372)

Architecture: Volumetric Residual UNet.
  - v1 "baseline":    1-channel input (defective skull only).
  - v3_full / v3_nosym: 2-channel input (defective skull + defect-type encoding),
    trained on all 5 SkullBreak defect types (DIM-88).

Output: Predicted implant volume (256^3 binary, soft probabilities)

Internally called "cran-2" in the CrAInial application.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# SkullBreak defect-type ordering must match scripts/train_wodzinski_v3.py::DEFECT_TYPES
# so the normalized channel encoding matches what the model was trained on.
DEFECT_TYPES = ["bilateral", "frontoorbital", "parietotemporal", "random_1", "random_2"]
DEFECT_TYPE_MAP = {dt: i for i, dt in enumerate(DEFECT_TYPES)}


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
    """Volumetric Residual UNet — Wodzinski et al. 2024 baseline (v1).

    BatchNorm + ReLU, sigmoid on output, 1-channel input.
    """

    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        self.enc1 = EncoderBlock(in_channels, base_filters)
        self.enc2 = EncoderBlock(base_filters, base_filters * 2)
        self.enc3 = EncoderBlock(base_filters * 2, base_filters * 4)
        self.enc4 = EncoderBlock(base_filters * 4, base_filters * 8)
        self.bottleneck = ResidualBlock3D(base_filters * 8, base_filters * 16)
        self.dec4 = DecoderBlock(base_filters * 16, base_filters * 8, base_filters * 8)
        self.dec3 = DecoderBlock(base_filters * 8, base_filters * 4, base_filters * 4)
        self.dec2 = DecoderBlock(base_filters * 4, base_filters * 2, base_filters * 2)
        self.dec1 = DecoderBlock(base_filters * 2, base_filters, base_filters)
        self.out_conv = nn.Conv3d(base_filters, 1, 1)
        self.sigmoid = nn.Sigmoid()

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
        return self.sigmoid(self.out_conv(d1))


# ======================= v3 Architecture (DIM-88) =======================
# InstanceNorm + LeakyReLU + dropout, raw-logits output; mirrors
# scripts/train_wodzinski_v3.py exactly so the trained state_dict loads clean.


class ResidualBlock3DV3(nn.Module):
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


class EncoderBlockV3(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ResidualBlock3DV3(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        s = self.block(x)
        p = self.pool(s)
        return s, p


class DecoderBlockV3(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 2, stride=2)
        self.block = ResidualBlock3DV3(in_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualUNet3DV3(nn.Module):
    """v3_full / v3_nosym architecture — DIM-88.

    Differs from baseline: InstanceNorm + LeakyReLU + dropout, 2-channel input
    (defective volume + defect-type encoding), raw logits output (apply sigmoid
    externally for probability maps).
    """

    def __init__(self, in_channels=2, base_filters=32, dropout=0.1):
        super().__init__()
        bf = base_filters
        self.enc1 = EncoderBlockV3(in_channels, bf)
        self.enc2 = EncoderBlockV3(bf, bf * 2, dropout)
        self.enc3 = EncoderBlockV3(bf * 2, bf * 4, dropout)
        self.enc4 = EncoderBlockV3(bf * 4, bf * 8, dropout)
        self.bottleneck = ResidualBlock3DV3(bf * 8, bf * 16, dropout)
        self.dec4 = DecoderBlockV3(bf * 16, bf * 8, bf * 8, dropout)
        self.dec3 = DecoderBlockV3(bf * 8, bf * 4, bf * 4, dropout)
        self.dec2 = DecoderBlockV3(bf * 4, bf * 2, bf * 2)
        self.dec1 = DecoderBlockV3(bf * 2, bf, bf)
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
        return self.out_conv(d1)  # raw logits


# ======================= Inference =======================


def _infer_in_channels(state_dict):
    """Read the first Conv3d weight tensor to find how many input channels the checkpoint was trained with."""
    for key in ("enc1.block.conv.0.weight", "module.enc1.block.conv.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    for key, tensor in state_dict.items():
        if key.endswith("enc1.block.conv.0.weight"):
            return int(tensor.shape[1])
    raise RuntimeError("Could not infer in_channels from checkpoint — unexpected layer naming.")


def _detect_architecture(state_dict):
    """Return 'v3' when state_dict uses InstanceNorm (6-layer residual block with Conv at index 4),
    otherwise 'baseline' (BatchNorm, 5-layer residual block)."""
    for key, tensor in state_dict.items():
        if key.endswith("enc1.block.conv.4.weight"):
            # v3: tensor is a 5-D conv weight; baseline: BN weight (1-D)
            return "v3" if tensor.ndim == 5 else "baseline"
    raise RuntimeError("Could not detect architecture — missing enc1.block.conv.4.weight.")


def load_wodzinski_model(checkpoint_path, device, base_filters=32, in_channels=None):
    """Load a trained Wodzinski ResidualUNet3D (baseline or v3) from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: torch.device to load model on.
        base_filters: Must match training config (32 for published baseline and v3_full).
        in_channels: Expected input channel count. If None, it is inferred from the
            checkpoint's first Conv3d weight so baseline (1ch) and v3 (2ch) both load.

    Returns:
        (model, in_channels): Model in eval mode on device, plus the channel
            count the caller must feed at inference time.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format at {checkpoint_path}")

    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    detected = _infer_in_channels(state_dict)
    if in_channels is None:
        in_channels = detected
    elif in_channels != detected:
        raise RuntimeError(
            f"Requested in_channels={in_channels} but checkpoint was trained with {detected}."
        )

    arch = _detect_architecture(state_dict)
    if arch == "v3":
        model = ResidualUNet3DV3(in_channels=in_channels, base_filters=base_filters, dropout=0.0)
        # Tag so inference knows to apply sigmoid externally (v3 outputs raw logits).
        model._wodzinski_variant = "v3"  # type: ignore[attr-defined]
    else:
        model = ResidualUNet3D(in_channels=in_channels, base_filters=base_filters)
        model._wodzinski_variant = "baseline"  # type: ignore[attr-defined]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, in_channels


def _encode_defect_type(defect_type):
    """Normalize defect_type string to the 0.0–1.0 scalar the model was trained with."""
    if defect_type is None:
        return None
    key = str(defect_type).strip().lower()
    if key not in DEFECT_TYPE_MAP:
        raise ValueError(
            f"Unknown defect_type={defect_type!r}; expected one of {DEFECT_TYPES}."
        )
    return DEFECT_TYPE_MAP[key] / (len(DEFECT_TYPES) - 1)


def preprocess_volume(volume, target_resolution=256, defect_type=None, in_channels=1):
    """Preprocess a volume for Wodzinski model input.

    Args:
        volume: numpy array (H, W, D) — defective skull binary volume.
        target_resolution: Target resolution (default 256).
        defect_type: Optional SkullBreak defect-type label. Required when
            in_channels >= 2; ignored for the 1-channel baseline.
        in_channels: Expected input channel count (1 for baseline, 2 for v3).

    Returns:
        tensor: (1, in_channels, R, R, R) float tensor on CPU.
    """
    from scipy.ndimage import zoom

    vol = (volume > 0).astype(np.float32)

    if vol.shape != (target_resolution, target_resolution, target_resolution):
        factors = [target_resolution / s for s in vol.shape]
        vol = zoom(vol, factors, order=1)

    channels = [vol]
    if in_channels >= 2:
        encoded = _encode_defect_type(defect_type)
        if encoded is None:
            raise ValueError(
                "Model requires defect_type (2-channel input) but none was provided."
            )
        channels.append(np.full_like(vol, encoded, dtype=np.float32))

    if len(channels) != in_channels:
        raise ValueError(
            f"Built {len(channels)} input channels but model expects {in_channels}."
        )

    stacked = np.stack(channels, axis=0)  # (C, R, R, R)
    tensor = torch.from_numpy(stacked).unsqueeze(0).float()  # (1, C, R, R, R)
    return tensor


def postprocess_output(output_tensor, threshold=0.5, is_logits=False):
    """Postprocess model output to binary implant volume.

    Args:
        output_tensor: (1, 1, R, R, R) soft prediction or raw logits.
        threshold: Binarization threshold (default 0.5, applied to probabilities).
        is_logits: If True, apply sigmoid before thresholding (v3 convention).

    Returns:
        volume: numpy array (R, R, R) uint8 binary volume.
    """
    if is_logits:
        output_tensor = torch.sigmoid(output_tensor)
    pred = output_tensor.squeeze().cpu().numpy()
    return (pred > threshold).astype(np.uint8)


def run_wodzinski_inference(model, volume, device, threshold=0.5, defect_type=None):
    """Run Wodzinski inference on a defective skull volume.

    Args:
        model: Loaded ResidualUNet3D model in eval mode.
        volume: numpy array (H, W, D) — defective skull binary volume.
        device: torch.device.
        threshold: Binarization threshold.
        defect_type: SkullBreak defect-type label, required when the model was
            trained with a defect-type channel (v3_full / v3_nosym).

    Returns:
        implant_volume: numpy uint8 array (256, 256, 256) — predicted implant.
        inference_time: float — inference time in seconds.
    """
    import time

    in_channels = int(model.enc1.block.conv[0].in_channels)
    input_tensor = preprocess_volume(
        volume, defect_type=defect_type, in_channels=in_channels
    ).to(device)

    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_time = time.time() - start

    is_logits = getattr(model, "_wodzinski_variant", "baseline") == "v3"
    implant_volume = postprocess_output(output, threshold=threshold, is_logits=is_logits)
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
