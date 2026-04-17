#!/usr/bin/env python3
"""Inference Pareto benchmark: pcdiff vs wv3 vs RF on the SkullBreak test subset.

Deliverable for DIM-96. Produces a reusable, parameterised harness that:
  * selects a seeded, stratified 20-case subset from `datasets/SkullBreak/test.csv`,
  * runs each method across its NFE sweep with one discarded warmup per sweep point,
  * measures wall-clock on GPU (`torch.cuda.synchronize` fenced, file I/O excluded),
  * computes DSC / bDSC_10mm / HD95 / ASSD on the final volume,
  * writes per-case CSV + per-(method, nfe) summary JSON + canonical stage report.

Method sweeps follow DIM-96:
  pcdiff : {50, 100, 250, 500, 1000}    (DPM-Solver++ for the <=500 points, DDPM for 1000)
  wv3    : single inference, no NFE knob
  RF     : {1, 2, 4, 8, 16}

pcdiff and RF both produce point clouds; they share the voxelization post-process so
raw-PC time and end-to-end-with-postproc time are reported separately. wv3 is volumetric
and has only a single end-to-end number.

The Pareto plot is produced by `benchmarking/pareto_plot.py` from this harness's CSV.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "pcdiff", REPO_ROOT / "voxelization"):
    p = str(_path)
    if p not in sys.path:
        sys.path.insert(0, p)

import diplib as dip  # noqa: E402
import nrrd  # noqa: E402

from benchmarking.reporting import (  # noqa: E402
    build_stage_report,
    summarize_numeric_fields,
    utc_now_iso,
    write_csv,
    write_json,
)
from voxelization.eval_metrics import assd, bdc, dc, hd95  # noqa: E402
from voxelization.src import config as vox_config  # noqa: E402
from voxelization.src.model import Encode2Points  # noqa: E402
from voxelization.src.utils import (  # noqa: E402
    filter_voxels_within_radius,
    load_config,
    load_model_manual,
)
from run_skullbreak_eval import (  # noqa: E402
    DEFECT_TYPES,
    SampleInfo,
    prepare_predicted_implant_volume,
)


def load_skullbreak_cases(csv_path: Path) -> List[SampleInfo]:
    """Parse the canonical SkullBreak test.csv (columns: defective, implant, defect_type, name).

    Accepts either absolute paths or paths relative to the repo root.
    `read_dataset` in run_skullbreak_eval.py expects a different, complete-skull-path CSV
    format and gates on precomputed vox files we do not need here.
    """
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    samples: List[SampleInfo] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            defect = row["defect_type"].strip()
            if defect not in DEFECT_TYPES:
                continue
            name = row["name"].strip()
            # Strip any `_surf` suffix so case_id is the stable numeric id used in the repo.
            case_id = name[:-5] if name.endswith("_surf") else name
            defective_npy = Path(row["defective"].strip())
            implant_npy = Path(row["implant"].strip())
            for p in (defective_npy, implant_npy):
                if not p.is_absolute():
                    # relative entries in test.csv are repo-root relative
                    pass
            defective_npy = defective_npy if defective_npy.is_absolute() else (REPO_ROOT / defective_npy)
            implant_npy = implant_npy if implant_npy.is_absolute() else (REPO_ROOT / implant_npy)
            defective_nrrd = defective_npy.with_name(f"{case_id}.nrrd")
            implant_nrrd = implant_npy.with_name(f"{case_id}.nrrd")
            complete_path = REPO_ROOT / "datasets" / "SkullBreak" / "complete_skull" / f"{case_id}_surf.npy"

            if not defective_npy.exists() or not defective_nrrd.exists():
                continue
            if not implant_nrrd.exists():
                continue

            samples.append(
                SampleInfo(
                    case_id=case_id,
                    defect=defect,
                    complete_path=complete_path,
                    defective_npy=defective_npy,
                    defective_nrrd=defective_nrrd,
                    implant_npy=implant_npy,
                    implant_nrrd=implant_nrrd,
                    vox_pointcloud=Path(""),   # not used by this harness
                    vox_grid=Path(""),
                )
            )
    if not samples:
        raise RuntimeError(f"No valid samples discovered in {csv_path}")
    return samples

CSV_FIELDS = [
    "method",
    "nfe",
    "solver",
    "weights_variant",
    "case_id",
    "defect",
    "dice",
    "bdice_10mm",
    "hd95_mm",
    "assd_mm",
    "t_model_sec",
    "t_postproc_sec",
    "t_end_to_end_sec",
    "gpu_peak_memory_mb",
    "status",
    "error",
]

SUMMARY_FIELDS = [
    "dice",
    "bdice_10mm",
    "hd95_mm",
    "assd_mm",
    "t_model_sec",
    "t_postproc_sec",
    "t_end_to_end_sec",
    "gpu_peak_memory_mb",
]


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------


def stratified_subset(
    samples: Sequence[SampleInfo],
    num_cases: int,
    seed: int,
) -> List[SampleInfo]:
    """Stratified sample: even per-defect-type coverage, deterministic across runs."""
    if num_cases % len(DEFECT_TYPES) != 0:
        raise ValueError(
            f"num_cases={num_cases} must be divisible by len(DEFECT_TYPES)="
            f"{len(DEFECT_TYPES)} for an even stratum."
        )
    per_defect = num_cases // len(DEFECT_TYPES)
    rng = np.random.default_rng(seed)
    buckets: Dict[str, List[SampleInfo]] = {d: [] for d in DEFECT_TYPES}
    for s in samples:
        if s.defect in buckets:
            buckets[s.defect].append(s)
    picked: List[SampleInfo] = []
    for defect in DEFECT_TYPES:
        pool = buckets[defect]
        if len(pool) < per_defect:
            raise RuntimeError(
                f"Only {len(pool)} cases available for defect '{defect}', need {per_defect}."
            )
        idx = rng.choice(len(pool), size=per_defect, replace=False)
        for i in sorted(idx):
            picked.append(pool[int(i)])
    return picked


# ---------------------------------------------------------------------------
# Timing + memory helpers
# ---------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> Tuple[Any, float]:
    _sync(device)
    t0 = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - t0


def peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    dice: float
    bdice_10mm: float
    hd95_mm: float
    assd_mm: float
    t_model_sec: float
    t_postproc_sec: float
    t_end_to_end_sec: float
    gpu_peak_memory_mb: float
    status: str = "ok"
    error: str = ""


def compute_metrics(
    pred_implant: np.ndarray,
    gt_implant: np.ndarray,
    defective_volume: np.ndarray,
    voxel_spacing: np.ndarray,
) -> Tuple[float, float, float, float]:
    dice = float(dc(pred_implant, gt_implant))
    bdice = float(
        bdc(pred_implant, gt_implant, defective_volume, voxelspacing=voxel_spacing, distance=10)
    )
    try:
        haus95 = float(hd95(pred_implant, gt_implant, voxelspacing=voxel_spacing))
    except RuntimeError:
        haus95 = float("nan")
    try:
        a_ssd = float(assd(pred_implant, gt_implant, voxelspacing=voxel_spacing))
    except RuntimeError:
        a_ssd = float("nan")
    return dice, bdice, haus95, a_ssd


class PointCloudPostprocessor:
    """Shared voxelization + morph post-processing for pcdiff and RF outputs."""

    def __init__(self, vox_config_path: Path, vox_checkpoint: Path, device: torch.device):
        default_config = REPO_ROOT / "voxelization" / "configs" / "default.yaml"
        cfg = load_config(str(vox_config_path.resolve()), str(default_config))
        cfg["test"]["model_file"] = str(vox_checkpoint.resolve())
        self.cfg = cfg
        self.device = device
        self.model = Encode2Points(cfg).to(device)
        state_dict = torch.load(cfg["test"]["model_file"], map_location="cpu")
        load_model_manual(state_dict["state_dict"], self.model)
        self.model.eval()
        self.generator = vox_config.get_generator(self.model, cfg, device=device)

    def to_implant_volume(
        self,
        defective_points: np.ndarray,
        implant_points: np.ndarray,
        defective_volume: np.ndarray,
    ) -> np.ndarray:
        combined = np.concatenate([defective_points, implant_points], axis=0) / 512.0
        inputs = torch.from_numpy(combined).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, _, _, psr_grid = self.generator.generate_mesh(inputs)
        psr_grid_np = psr_grid.detach().cpu().numpy()[0]
        return prepare_predicted_implant_volume(
            psr_grid_np, defective_volume, implant_points / 512.0
        )


class PCDiffMethod:
    """pcdiff + voxelization. NFE via DPM-Solver++ (<=500) or DDPM (1000)."""

    name = "pcdiff"

    def __init__(
        self,
        checkpoint: Path,
        postprocessor: PointCloudPostprocessor,
        device: torch.device,
        num_points: int = 30720,
        num_nn: int = 3072,
        weights_variant: str = "raw",
    ):
        from run_skullbreak_eval import PCDiffRunner  # defer to avoid double-import cost

        self.device = device
        self.num_points = num_points
        self.num_nn = num_nn
        self.checkpoint = checkpoint
        self.post = postprocessor
        self.weights_variant = weights_variant
        # Inner runner is rebuilt per-NFE because sampling_method/steps are constructor-bound.
        self._PCDiffRunner = PCDiffRunner
        self._runner: Optional[Any] = None
        self._current_nfe: Optional[int] = None
        self._current_solver: Optional[str] = None

    def nfe_sweep(self) -> List[Tuple[int, str]]:
        return [(50, "dpm_solver"), (100, "dpm_solver"), (250, "dpm_solver"),
                (500, "dpm_solver"), (1000, "ddpm")]

    def prepare_nfe(self, nfe: int, solver: str) -> None:
        if self._runner is not None and self._current_nfe == nfe and self._current_solver == solver:
            return
        del self._runner
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self._runner = self._PCDiffRunner(
            checkpoint=self.checkpoint,
            device=self.device,
            num_points=self.num_points,
            num_nn=self.num_nn,
            sampling_method=solver,
            sampling_steps=nfe,
        )
        self._current_nfe = nfe
        self._current_solver = solver

    def run_case(self, sample: SampleInfo) -> CaseResult:
        defective_points = np.load(sample.defective_npy).astype(np.float32)
        defective_vol, header = nrrd.read(str(sample.defective_nrrd))
        defective_vol = defective_vol.astype(np.uint8)
        gt_implant, _ = nrrd.read(str(sample.implant_nrrd))
        gt_implant = gt_implant.astype(np.uint8)
        spacing = _nrrd_spacing(header)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        (implant_points, _implant_norm, _shift, _scale), t_model = timed(
            self.device, lambda: self._runner.generate_implant(defective_points)
        )
        pred_implant, t_post = timed(
            self.device,
            lambda: self.post.to_implant_volume(defective_points, implant_points, defective_vol),
        )
        t_total = t_model + t_post
        peak_mb = peak_memory_mb(self.device)
        d, b, h, a = compute_metrics(pred_implant, gt_implant, defective_vol, spacing)
        return CaseResult(d, b, h, a, t_model, t_post, t_total, peak_mb)


class Wv3Method:
    """Wodzinski v3 (no-sym) — direct volumetric, single inference per case."""

    name = "wv3"

    def __init__(self, checkpoint: Path, device: torch.device, weights_variant: str = "raw"):
        from runpod_serverless.wodzinski_inference import (  # noqa: E402
            load_wodzinski_model,
            preprocess_volume,
            postprocess_output,
        )

        self.device = device
        self.checkpoint = checkpoint
        self.weights_variant = weights_variant
        self.model = load_wodzinski_model(str(checkpoint.resolve()), device)
        self._pre = preprocess_volume
        self._post = postprocess_output

    def nfe_sweep(self) -> List[Tuple[int, str]]:
        return [(1, "direct")]

    def prepare_nfe(self, nfe: int, solver: str) -> None:
        return  # single-shot volumetric, no setup

    def run_case(self, sample: SampleInfo) -> CaseResult:
        defective_vol, header = nrrd.read(str(sample.defective_nrrd))
        defective_vol_u8 = defective_vol.astype(np.uint8)
        gt_implant, _ = nrrd.read(str(sample.implant_nrrd))
        gt_implant = gt_implant.astype(np.uint8)
        spacing = _nrrd_spacing(header)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        input_tensor = self._pre(defective_vol_u8).to(self.device)

        def _infer():
            with torch.no_grad():
                return self.model(input_tensor)

        output, t_model = timed(self.device, _infer)
        pred_at_256 = self._post(output, threshold=0.5)

        # wv3 predicts at 256^3. The product deliverable is a native-resolution implant, so
        # count the upsample in end-to-end wall-clock (nearest-neighbor to preserve the
        # predicted binary mask exactly — trilinear + threshold would introduce sub-voxel
        # boundary artifacts that bias HD95/ASSD against wv3).
        from scipy.ndimage import zoom

        factors = [s / t for s, t in zip(defective_vol_u8.shape, pred_at_256.shape)]

        def _upsample():
            return (zoom(pred_at_256.astype(np.float32), factors, order=0) > 0.5).astype(np.uint8)

        pred_full, t_post = timed(self.device, _upsample)
        peak_mb = peak_memory_mb(self.device)
        d, b, h, a = compute_metrics(pred_full, gt_implant, defective_vol_u8, spacing)
        return CaseResult(d, b, h, a, t_model, t_post, t_model + t_post, peak_mb)


class RFMethod:
    """Rectified Flow (epoch-3 checkpoint) + voxelization. NFE = Euler steps."""

    name = "rf"

    def __init__(
        self,
        checkpoint: Path,
        postprocessor: PointCloudPostprocessor,
        device: torch.device,
        num_points: int = 30720,
        num_nn: int = 3072,
        weights_variant: str = "ema",
    ):
        from pcdiff.train_completion import PVCNN2  # noqa: E402
        from rectified_flow import RectifiedFlow  # noqa: E402

        self.device = device
        self.num_points = num_points
        self.num_nn = num_nn
        self.sv_points = num_points - num_nn
        self.post = postprocessor
        self.weights_variant = weights_variant

        ckpt = torch.load(str(checkpoint.resolve()), map_location="cpu")
        ckpt_args = ckpt.get("args", {})
        embed_dim = int(ckpt_args.get("embed_dim", 64))
        width_mult = float(ckpt_args.get("width_mult", 1.0))

        self.model = PVCNN2(
            use_att=True,
            dropout=0.1,
            extra_feature_channels=0,
            num_classes=3,
            embed_dim=embed_dim,
            width_multiplier=width_mult,
            voxel_resolution_multiplier=1,
            sv_points=self.sv_points,
        ).to(device)
        key = "ema_state_dict" if (weights_variant == "ema" and "ema_state_dict" in ckpt) else "state_dict"
        state = ckpt[key]
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.weights_variant = key.replace("_state_dict", "")

        self.rf = RectifiedFlow(sv_points=self.sv_points)
        self._num_steps: Optional[int] = None

    def nfe_sweep(self) -> List[Tuple[int, str]]:
        return [(1, "euler"), (2, "euler"), (4, "euler"), (8, "euler"), (16, "euler")]

    def prepare_nfe(self, nfe: int, solver: str) -> None:
        self._num_steps = nfe

    def _generate_implant(self, defective_points: np.ndarray) -> np.ndarray:
        assert self._num_steps is not None
        idx = np.random.choice(defective_points.shape[0], self.sv_points, replace=False)
        partial_raw = defective_points[idx]
        pc_min, pc_max = partial_raw.min(axis=0), partial_raw.max(axis=0)
        shift = (pc_min + pc_max) / 2.0
        scale = (pc_max - pc_min).max() / 2.0 / 3.0
        if scale <= 0:
            raise ValueError("Invalid scale derived from defective point cloud.")
        partial = (partial_raw - shift) / scale
        partial_t = torch.from_numpy(partial.astype(np.float32)).unsqueeze(0).transpose(1, 2).to(self.device)
        shape = (1, 3, self.num_nn)
        with torch.no_grad():
            combined = self.rf.euler_sample(self.model, partial_t, shape, self.device, num_steps=self._num_steps)
        implant_norm = combined[:, :, self.sv_points:].transpose(1, 2)[0].cpu().numpy()
        return (implant_norm * scale + shift).astype(np.float32)

    def run_case(self, sample: SampleInfo) -> CaseResult:
        defective_points = np.load(sample.defective_npy).astype(np.float32)
        defective_vol, header = nrrd.read(str(sample.defective_nrrd))
        defective_vol = defective_vol.astype(np.uint8)
        gt_implant, _ = nrrd.read(str(sample.implant_nrrd))
        gt_implant = gt_implant.astype(np.uint8)
        spacing = _nrrd_spacing(header)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        implant_points, t_model = timed(self.device, lambda: self._generate_implant(defective_points))
        pred_implant, t_post = timed(
            self.device,
            lambda: self.post.to_implant_volume(defective_points, implant_points, defective_vol),
        )
        peak_mb = peak_memory_mb(self.device)
        d, b, h, a = compute_metrics(pred_implant, gt_implant, defective_vol, spacing)
        return CaseResult(d, b, h, a, t_model, t_post, t_model + t_post, peak_mb)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _nrrd_spacing(header: Dict[str, Any]) -> np.ndarray:
    sd = header["space directions"]
    return np.asarray([sd[0, 0], sd[1, 1], sd[2, 2]])


def run_sweep(
    method,
    cases: Sequence[SampleInfo],
    output_rows: List[Dict[str, Any]],
    log: Callable[[str], None],
) -> None:
    for nfe, solver in method.nfe_sweep():
        method.prepare_nfe(nfe, solver)
        log(f"[{method.name}] nfe={nfe} solver={solver}: warmup on {cases[0].case_id}")
        try:
            method.run_case(cases[0])  # warmup, discarded
        except Exception as exc:  # noqa: BLE001
            log(f"[{method.name}] warmup failed: {exc}")

        for sample in cases:
            try:
                result = method.run_case(sample)
                log(
                    f"[{method.name} nfe={nfe}] {sample.case_id}/{sample.defect}: "
                    f"DSC={result.dice:.4f} HD95={result.hd95_mm:.2f} "
                    f"ASSD={result.assd_mm:.2f} t_model={result.t_model_sec:.3f}s "
                    f"t_total={result.t_end_to_end_sec:.3f}s"
                )
            except Exception as exc:  # noqa: BLE001
                log(f"[{method.name} nfe={nfe}] {sample.case_id} FAILED: {exc}")
                result = CaseResult(
                    dice=float("nan"), bdice_10mm=float("nan"), hd95_mm=float("nan"),
                    assd_mm=float("nan"), t_model_sec=float("nan"), t_postproc_sec=float("nan"),
                    t_end_to_end_sec=float("nan"), gpu_peak_memory_mb=float("nan"),
                    status="error", error=str(exc),
                )
            row = asdict(result)
            row.update({
                "method": method.name,
                "nfe": nfe,
                "solver": solver,
                "weights_variant": getattr(method, "weights_variant", "raw"),
                "case_id": sample.case_id,
                "defect": sample.defect,
            })
            output_rows.append(row)


def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        groups.setdefault((r["method"], int(r["nfe"]), r["solver"]), []).append(r)

    summaries: List[Dict[str, Any]] = []
    for (method, nfe, solver), sub in sorted(groups.items()):
        stats = summarize_numeric_fields(sub, SUMMARY_FIELDS)
        summaries.append({
            "method": method,
            "nfe": nfe,
            "solver": solver,
            "weights_variant": sub[0].get("weights_variant", "raw"),
            "n_cases": len(sub),
            "metrics": stats,
        })
    return summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pcdiff / wv3 / RF inference Pareto benchmark (DIM-96).")
    p.add_argument("--dataset-csv", default="datasets/SkullBreak/test.csv")
    p.add_argument("--output-dir", required=True, help="Directory for CSV + JSON + stage report.")
    p.add_argument("--num-cases", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--methods", nargs="+", default=["pcdiff", "wv3", "rf"],
                   choices=["pcdiff", "wv3", "rf"])
    p.add_argument("--pcdiff-checkpoint", default=None)
    p.add_argument("--wv3-checkpoint", default=None)
    p.add_argument("--rf-checkpoint", default=None,
                   help="Path to RF epoch-3 checkpoint (loss=2.214).")
    p.add_argument("--vox-checkpoint", default="voxelization/checkpoints/model_best.pt")
    p.add_argument("--vox-config", default="voxelization/configs/gen_skullbreak.yaml")
    p.add_argument("--rf-weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--pcdiff-weights", choices=["ema", "raw"], default="raw",
                   help="pcdiff checkpoint stores only `model_state`; document if EMA added later.")
    p.add_argument("--wv3-weights", choices=["ema", "raw"], default="raw")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now_iso()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset_csv = Path(args.dataset_csv)
    if not dataset_csv.is_absolute():
        dataset_csv = (REPO_ROOT / dataset_csv).resolve()
    all_samples = load_skullbreak_cases(dataset_csv)
    cases = stratified_subset(all_samples, args.num_cases, args.seed)

    manifest_rows = [
        {"order": i, "case_id": c.case_id, "defect": c.defect, "defective_nrrd": str(c.defective_nrrd)}
        for i, c in enumerate(cases)
    ]
    write_json(output_dir / "case_manifest.json", {
        "seed": args.seed,
        "num_cases": args.num_cases,
        "dataset_csv": str(dataset_csv),
        "cases": manifest_rows,
    })

    log_path = output_dir / "pareto_bench.log"
    log_handle = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    log(f"device={device} cases={len(cases)} seed={args.seed} methods={args.methods}")

    needs_pp = ("pcdiff" in args.methods) or ("rf" in args.methods)
    postprocessor: Optional[PointCloudPostprocessor] = None
    if needs_pp:
        vox_ckpt = Path(args.vox_checkpoint)
        if not vox_ckpt.is_absolute():
            vox_ckpt = REPO_ROOT / vox_ckpt
        vox_cfg = Path(args.vox_config)
        if not vox_cfg.is_absolute():
            vox_cfg = REPO_ROOT / vox_cfg
        log(f"Loading shared voxelization post-processor from {vox_ckpt}")
        postprocessor = PointCloudPostprocessor(vox_cfg, vox_ckpt, device)

    rows: List[Dict[str, Any]] = []
    if "pcdiff" in args.methods:
        if not args.pcdiff_checkpoint:
            raise SystemExit("--pcdiff-checkpoint is required when pcdiff is in --methods.")
        method = PCDiffMethod(Path(args.pcdiff_checkpoint), postprocessor, device,
                              weights_variant=args.pcdiff_weights)
        run_sweep(method, cases, rows, log)
        del method
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "wv3" in args.methods:
        if not args.wv3_checkpoint:
            raise SystemExit("--wv3-checkpoint is required when wv3 is in --methods.")
        method = Wv3Method(Path(args.wv3_checkpoint), device, weights_variant=args.wv3_weights)
        run_sweep(method, cases, rows, log)
        del method
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "rf" in args.methods:
        if not args.rf_checkpoint:
            raise SystemExit("--rf-checkpoint is required when rf is in --methods.")
        method = RFMethod(Path(args.rf_checkpoint), postprocessor, device,
                          weights_variant=args.rf_weights)
        run_sweep(method, cases, rows, log)
        del method
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = output_dir / "pareto_cases.csv"
    write_csv(csv_path, rows, CSV_FIELDS)

    summaries = summarize_rows(rows)
    write_json(output_dir / "pareto_summary.json", {
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "seed": args.seed,
        "methods": args.methods,
        "groups": summaries,
    })

    report = build_stage_report(
        stage_name="pareto_inference_benchmark",
        dataset="SkullBreak",
        repo_root=REPO_ROOT,
        started_at=started_at,
        finished_at=utc_now_iso(),
        command=sys.argv,
        args=vars(args),
        outputs={
            "cases_csv": str(csv_path),
            "summary_json": str(output_dir / "pareto_summary.json"),
            "case_manifest": str(output_dir / "case_manifest.json"),
            "log": str(log_path),
        },
        summary={"n_rows": len(rows), "groups": len(summaries)},
        device=device,
    )
    write_json(output_dir / "pareto_stage_report.json", report)
    log(f"Wrote {csv_path}, summary, and stage report.")
    log_handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
