import argparse
import os
import sys
import time
from pathlib import Path

import diplib as dip
import nrrd
import numpy as np
import torch
import torch.distributed as dist
import yaml
from scipy import ndimage
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval_metrics import bdc, dc, hd95
from src import config
from src.data.core import SkullEval
from src.model import Encode2Points
from src.postprocess import build_refined_complete_mask, select_implant_region
from src.utils import (
    crop,
    load_config,
    load_model_manual,
    padding,
    re_sample_shape,
    readCT,
    reverse_crop,
    reverse_padding,
)

from benchmarking.reporting import build_stage_report, summarize_numeric_fields, utc_now_iso, write_csv, write_json

np.set_printoptions(precision=4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file you want to use.")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disable CUDA even if available")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="Random seed")
    parser.add_argument("--iter", type=int, metavar="S", help="Training iteration to evaluate (unused)")
    parser.add_argument("--dist-backend", default="nccl", help="torch.distributed backend")
    return parser.parse_args()


def init_distributed(backend: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_dist = world_size > 1
    if is_dist:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    return is_dist, world_size, rank, local_rank


def cleanup_distributed(is_dist: bool):
    if is_dist and dist.is_initialized():
        dist.destroy_process_group()


def gather_case_records(case_records, world_size, is_dist):
    if not is_dist:
        return case_records
    gathered = [None] * world_size
    dist.all_gather_object(gathered, case_records)
    merged = []
    for chunk in gathered:
        merged.extend(chunk or [])
    return merged


def main():
    args = parse_args()
    stage_started_at = utc_now_iso()
    is_dist, world_size, rank, local_rank = init_distributed(args.dist_backend)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {args.config}\nProvide a valid YAML config path as the first argument."
        )
    default_config = str(REPO_ROOT / "voxelization" / "configs" / "default.yaml")
    cfg = load_config(args.config, default_config)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    device_index = local_rank if use_cuda else None

    out_dir = Path(cfg["train"]["out_dir"] or "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    generation_dir = out_dir / cfg["generation"]["generation_dir"]
    generation_dir.mkdir(parents=True, exist_ok=True)

    data_path = cfg["data"]["path"]
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Dataset path does not exist: {data_path}\n"
            f"Set the correct path in your config or via the PCDIFF_SKULL*_RESULTS env var."
        )
    dataset = SkullEval(data_path)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if is_dist else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["generation"].get("batch_size", 1),
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg["generation"].get("num_workers", 0),
        pin_memory=use_cuda,
    )

    model = Encode2Points(cfg).to(device)
    if is_dist and use_cuda:
        ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
        module = ddp_model.module
    else:
        ddp_model = model
        module = model

    model_file = cfg["test"]["model_file"]
    if not Path(model_file).is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_file}\n"
            f"Set the correct path in your config or via the PCDIFF_SKULL*_MODEL env var."
        )
    if rank == 0:
        print("\n---------- Load model----------")
        print("Load best model from: " + model_file + "\n")
    try:
        state_dict = torch.load(model_file, map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{model_file}': {e}") from e
    module = ddp_model.module if isinstance(ddp_model, DistributedDataParallel) else ddp_model
    load_model_manual(state_dict["state_dict"], module)

    generator = config.get_generator(module, cfg, device=device)
    ddp_model.eval()

    if rank == 0:
        print("\n---------- Voxelizing point clouds ----------")

    case_records = []

    for it, data in enumerate(loader):
        if rank == 0 and it % max(1, len(loader) // 10 or 1) == 0:
            print(f"Sampling step [{it + 1}/{len(loader)}] ...")
        name = data["name"][0]
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(device_index)
            torch.cuda.synchronize(device_index)
        case_start = time.perf_counter()

        # ----- Get defective skull -----
        if cfg["data"]["dset"] == "SkullBreak":
            defective_skull, header = nrrd.read(
                os.path.join(
                    name.split("/results")[0],
                    "defective_skull",
                    name.split("syn/")[1][:-8],
                    name.split("_surf")[0][-3:] + ".nrrd",
                )
            )

        if cfg["data"]["dset"] == "SkullFix":
            defective_skull = readCT(
                os.path.join(
                    name.split("/results")[0],
                    "defective_skull",
                    name.split("_surf")[0][-3:] + ".nrrd",
                )
            )
            defective_skull, idx_x, idx_y, idx_z, shape = crop(defective_skull)
            defective_skull, dim_x, dim_y, dim_z = padding(defective_skull)
            defective_skull = defective_skull.astype(np.float32)

        inputs = data["inputs"][0, :, :, :]
        defect_point_count = int(cfg["generation"].get("defect_point_count", 3072))
        # If SkullEval provided skull_point_count, extract defect points from
        # the skull portion (first N points) instead of the tail of the array.
        skull_pt_count = data.get("skull_point_count")
        if skull_pt_count is not None and skull_pt_count > 0:
            defect_points = (
                inputs[0, :skull_pt_count, :].detach().cpu().numpy() * 512
            )
        else:
            defect_points = inputs[0, max(inputs.shape[1] - defect_point_count, 0) :, :].detach().cpu().numpy() * 512
        completes = np.zeros((512, 512, 512))

        if cfg["generation"]["num_ensemble"] >= 2:
            for pc in tqdm(range(cfg["generation"]["num_ensemble"]), total=cfg["generation"]["num_ensemble"]):
                vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs[pc, :, :].unsqueeze(dim=0))

                psr_grid = psr_grid.detach().cpu().numpy()
                psr_grid = psr_grid[0, :, :, :]
                out = np.zeros((512, 512, 512))
                out[psr_grid <= 0] = 1
                out = ndimage.binary_dilation(out)

                completes += out

                if cfg["generation"]["save_ensemble_implants"]:
                    out = out - defective_skull
                    out = dip.MedianFilter(out, dip.Kernel(shape="rectangular", param=(3, 3, 3)))
                    out.Convert("BIN")
                    out = dip.Closing(out, dip.SE((3, 3, 3)))
                    out = dip.Closing(out, dip.SE((3, 3, 3)))
                    out = dip.FillHoles(out)
                    out = dip.Label(out, mode="largest")
                    out = np.asarray(out, dtype=np.float32)

                    if cfg["data"]["dset"] == "SkullFix":
                        util, header = nrrd.read(
                            os.path.join(
                                name.split("/results")[0],
                                "defective_skull",
                                name.split("_surf")[0][-3:] + ".nrrd",
                            )
                        )
                        out = reverse_padding(out, dim_x, dim_y, dim_z)
                        out = reverse_crop(out, idx_x, idx_y, idx_z, shape)

                        new_shape = np.asarray(util.shape)
                        out_re, _ = re_sample_shape(out, [0.45, 0.45, 0.45], new_shape)
                        im = np.zeros(out_re.shape)
                        im[out_re > 0.5] = 1
                        nrrd.write(str(name + "/impl_" + str(pc) + ".nrrd"), im, header)

                    if cfg["data"]["dset"] == "SkullBreak":
                        nrrd.write(str(name + "/impl_" + str(pc) + ".nrrd"), out, header)
        else:
            vertices, faces, points, normals, psr_grid = generator.generate_mesh(inputs[0, :, :].unsqueeze(dim=0))

            psr_grid = psr_grid.detach().cpu().numpy()
            psr_grid = psr_grid[0, :, :, :]
            out = np.zeros((512, 512, 512))
            out[psr_grid <= 0] = 1
            out = ndimage.binary_dilation(out)

            completes += out

        complete_prob = completes / max(1, cfg["generation"]["num_ensemble"])
        refinement_cfg = cfg["generation"]["refinement"]
        if refinement_cfg["strategy"] == "occupancy_symmetry":
            mean_complete = build_refined_complete_mask(
                complete_prob,
                defect_points,
                complete_prob.shape,
                threshold=refinement_cfg["occupancy_threshold"],
                symmetry_weight=refinement_cfg["symmetry_weight"],
                symmetry_axis=refinement_cfg["symmetry_axis"],
                symmetry_defect_only=refinement_cfg["symmetry_defect_only"],
                defect_region_method=cfg["generation"]["postprocess"]["method"],
                bbox_margin_voxels=cfg["generation"]["postprocess"]["bbox_margin_voxels"],
                radius_scale=cfg["generation"]["postprocess"]["radius_scale"],
            )
        elif refinement_cfg["strategy"] == "legacy":
            mean_complete = np.zeros((512, 512, 512))
            mean_complete[completes >= np.ceil(cfg["generation"]["num_ensemble"] / 2)] = 1
        else:
            raise ValueError(f"Unsupported refinement strategy: {refinement_cfg['strategy']}")

        mean_implant = mean_complete - defective_skull
        mean_implant = select_implant_region(
            defect_points,
            mean_implant,
            method=cfg["generation"]["postprocess"]["method"],
            bbox_margin_voxels=cfg["generation"]["postprocess"]["bbox_margin_voxels"],
            radius_scale=cfg["generation"]["postprocess"]["radius_scale"],
            fallback=cfg["generation"]["postprocess"]["fallback"],
        )
        mean_implant = mean_implant.astype(bool)
        mean_implant = dip.Opening(mean_implant, dip.SE((3, 3, 3)))
        mean_implant = dip.Label(mean_implant, mode="largest")
        mean_implant = dip.MedianFilter(mean_implant, dip.Kernel(shape="rectangular", param=(3, 3, 3)))
        mean_implant.Convert("BIN")
        mean_implant = dip.Closing(mean_implant, dip.SE((3, 3, 3)))
        mean_implant = dip.FillHoles(mean_implant)

        if cfg["data"]["dset"] == "SkullBreak":
            mean_implant = dip.Label(mean_implant, mode="largest")
            mean_implant = np.asarray(mean_implant, dtype=np.float32)
            gt_implant, _ = nrrd.read(
                os.path.join(
                    name.split("/results")[0],
                    "implant",
                    name.split("syn/")[1][:-8],
                    name.split("_surf")[0][-3:] + ".nrrd",
                )
            )
            defective_skull, header = nrrd.read(
                os.path.join(
                    name.split("/results")[0],
                    "defective_skull",
                    name.split("syn/")[1][:-8],
                    name.split("_surf")[0][-3:] + ".nrrd",
                )
            )

        if cfg["data"]["dset"] == "SkullFix":
            mean_implant = dip.Label(mean_implant, mode="largest")
            mean_implant = np.asarray(mean_implant, dtype=np.float32)
            gt_implant, _ = nrrd.read(
                os.path.join(name.split("/results")[0], "implant", name.split("_surf")[0][-3:] + ".nrrd")
            )
            defective_skull, header = nrrd.read(
                os.path.join(name.split("/results")[0], "defective_skull", name.split("_surf")[0][-3:] + ".nrrd")
            )

        new_shape = np.asarray(defective_skull.shape)
        spacing = np.asarray(
            [
                header["space directions"][0, 0],
                header["space directions"][1, 1],
                header["space directions"][2, 2],
            ]
        )

        if cfg["data"]["dset"] == "SkullFix":
            mean_implant = reverse_padding(mean_implant, dim_x, dim_y, dim_z)
            mean_implant = reverse_crop(mean_implant, idx_x, idx_y, idx_z, shape)
            mean_implant_re, _ = re_sample_shape(mean_implant, [0.45, 0.45, 0.45], new_shape)
            mean_implant = np.zeros(mean_implant_re.shape)
            mean_implant[mean_implant_re > 0.5] = 1
            mean_implant = mean_implant.astype(bool)
            mean_implant = dip.Label(mean_implant, mode="largest")
            mean_implant = np.asarray(mean_implant, dtype=np.float32)

        mean_implant_path = str(name + "/mean_impl.nrrd")
        nrrd.write(mean_implant_path, mean_implant, header)

        metrics = {
            "dice": None,
            "bdice_10mm": None,
            "hd95_mm": None,
        }
        eval_metrics_path = None

        if cfg["generation"]["compute_eval_metrics"]:
            eval_metrics_path = str(name + "/eval_metrics.yaml")
            eval_mets = {}

            print("Compute eval metrics for: " + name)
            print("Voxelspacing: " + str(spacing))

            dice = float(dc(mean_implant, gt_implant))
            metrics["dice"] = dice
            eval_mets["dice"] = dice
            print("Dice score: " + str(dice))

            bdice = float(bdc(mean_implant, gt_implant, defective_skull, voxelspacing=spacing))
            metrics["bdice_10mm"] = bdice
            eval_mets["bdice"] = bdice
            eval_mets["bdice_10mm"] = bdice
            print("Boundary dice (10mm): " + str(bdice))

            hausdorff95 = float(hd95(mean_implant, gt_implant, voxelspacing=spacing))
            metrics["hd95_mm"] = hausdorff95
            eval_mets["haussdorf95"] = hausdorff95
            eval_mets["hd95_mm"] = hausdorff95
            print("95 percentile Haussdorf distance: " + str(hausdorff95) + "\n")

            with open(eval_metrics_path, "w") as file:
                yaml.safe_dump(eval_mets, file, sort_keys=True)

        if use_cuda:
            torch.cuda.synchronize(device_index)
            peak_memory_mb = round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 2)
        else:
            peak_memory_mb = None
        runtime_sec = round(time.perf_counter() - case_start, 4)

        case_records.append(
            {
                "case_name": Path(name).name,
                "case_dir": name,
                "mean_implant_path": mean_implant_path,
                "eval_metrics_path": eval_metrics_path,
                "dice": metrics["dice"],
                "bdice_10mm": metrics["bdice_10mm"],
                "hd95_mm": metrics["hd95_mm"],
                "runtime_sec": runtime_sec,
                "gpu_peak_memory_mb": peak_memory_mb,
            }
        )

    all_case_records = sorted(
        gather_case_records(case_records, world_size, is_dist), key=lambda record: record["case_name"]
    )

    if rank == 0:
        summary = summarize_numeric_fields(
            all_case_records,
            ["dice", "bdice_10mm", "hd95_mm", "runtime_sec", "gpu_peak_memory_mb"],
        )
        summary.update(
            {
                "case_count": len(all_case_records),
                "num_ensemble": cfg["generation"]["num_ensemble"],
                "compute_eval_metrics": cfg["generation"]["compute_eval_metrics"],
            }
        )

        cases_csv_path = generation_dir / "benchmark_cases.csv"
        summary_json_path = generation_dir / "benchmark_summary.json"
        report_json_path = generation_dir / "benchmark_stage_report.json"

        write_csv(
            cases_csv_path,
            all_case_records,
            fieldnames=[
                "case_name",
                "case_dir",
                "mean_implant_path",
                "eval_metrics_path",
                "dice",
                "bdice_10mm",
                "hd95_mm",
                "runtime_sec",
                "gpu_peak_memory_mb",
            ],
        )
        write_json(summary_json_path, summary)
        write_json(
            report_json_path,
            build_stage_report(
                stage_name="voxelization-evaluation",
                dataset=cfg["data"]["dset"],
                repo_root=REPO_ROOT,
                started_at=stage_started_at,
                finished_at=utc_now_iso(),
                command=sys.argv,
                config=cfg,
                args=vars(args),
                outputs={
                    "generation_dir": str(generation_dir),
                    "cases_csv": str(cases_csv_path),
                    "summary_json": str(summary_json_path),
                    "data_path": cfg["data"]["path"],
                },
                summary=summary,
                extra={"model_checkpoint": cfg["test"]["model_file"]},
                device=device,
            ),
        )

    cleanup_distributed(is_dist)


if __name__ == "__main__":
    main()
