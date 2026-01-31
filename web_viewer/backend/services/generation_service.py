"""
Service for generating cranial implants using PCDiff + Voxelization pipeline.

This service orchestrates the end-to-end inference pipeline:
1. Load defective skull point cloud
2. Run PCDiff diffusion model to generate implant point cloud
3. Run voxelization to create watertight mesh
4. Save results as PointCloud records
"""

import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sqlalchemy.orm import Session

from web_viewer.backend.config import settings
from web_viewer.backend.models.generation_job import GenerationJob
from web_viewer.backend.models.notification import Notification
from web_viewer.backend.models.point_cloud import PointCloud
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.file_service import get_file_size

logger = logging.getLogger(__name__)

# Global model cache (loaded once, reused across jobs)
_model_cache: dict = {
    "pcdiff": None,
    "voxelization": None,
    "lock": threading.Lock(),
}


class GenerationService:
    """Service for managing implant generation jobs."""

    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    # ------------------------------------------------------------------
    # Job CRUD
    # ------------------------------------------------------------------

    def create_job(
        self,
        project_id: str,
        input_pc_id: str,
        sampling_method: str = "ddim",
        sampling_steps: int = 50,
        num_ensemble: int = 5,
        name: str | None = None,
        description: str | None = None,
    ) -> GenerationJob:
        """Create a new generation job in pending state."""
        # Validate input point cloud exists
        input_pc = self.db.query(PointCloud).filter(PointCloud.id == input_pc_id).first()
        if not input_pc:
            raise ValueError(f"Input point cloud not found: {input_pc_id}")

        # Generate default name if not provided
        if not name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = f"Generation_{input_pc.name}_{timestamp}"

        job = GenerationJob(
            name=name,
            description=description,
            project_id=project_id,
            input_pc_id=input_pc_id,
            status="pending",
            progress_percent=0,
            sampling_method=sampling_method,
            sampling_steps=sampling_steps,
            num_ensemble=num_ensemble,
            queued_at=datetime.now(timezone.utc),
        )
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        self.audit.log(
            action="generation_job.create",
            entity_type="generation_job",
            entity_id=job.id,
            details={
                "input_pc_id": input_pc_id,
                "sampling_method": sampling_method,
                "sampling_steps": sampling_steps,
                "num_ensemble": num_ensemble,
            },
        )

        # Create notification
        self._create_notification(
            type="generation_queued",
            title="Generation Queued",
            message=f"Implant generation '{name}' has been queued.",
            entity_type="generation_job",
            entity_id=job.id,
        )

        return job

    def get_job(self, job_id: str) -> GenerationJob | None:
        """Get a generation job by ID."""
        return self.db.query(GenerationJob).filter(GenerationJob.id == job_id).first()

    def list_jobs(
        self,
        project_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[GenerationJob]:
        """List generation jobs with optional filtering."""
        q = self.db.query(GenerationJob)
        if project_id:
            q = q.filter(GenerationJob.project_id == project_id)
        if status:
            q = q.filter(GenerationJob.status == status)
        return q.order_by(GenerationJob.created_at.desc()).offset(offset).limit(limit).all()

    def update_progress(
        self,
        job_id: str,
        progress_percent: int,
        current_step: str | None = None,
    ) -> GenerationJob | None:
        """Update job progress."""
        job = self.get_job(job_id)
        if not job:
            return None

        job.progress_percent = progress_percent
        if current_step:
            job.current_step = current_step

        self.db.commit()
        self.db.refresh(job)
        return job

    def cancel_job(self, job_id: str) -> GenerationJob | None:
        """Cancel a pending or running job."""
        job = self.get_job(job_id)
        if not job:
            return None

        if job.status in ("completed", "failed", "cancelled"):
            return job

        job.status = "cancelled"
        job.error_message = "Cancelled by user"
        job.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(job)

        self.audit.log(
            action="generation_job.cancel",
            entity_type="generation_job",
            entity_id=job_id,
        )

        return job

    def select_output(self, job_id: str, output_id: str) -> GenerationJob | None:
        """Select a specific output from ensemble results as the preferred one."""
        job = self.get_job(job_id)
        if not job:
            return None

        # Verify the output_id is in the job's outputs
        output_ids = json.loads(job.output_pc_ids_json) if job.output_pc_ids_json else []
        stl_ids = json.loads(job.output_stl_ids_json) if job.output_stl_ids_json else []
        all_output_ids = output_ids + stl_ids

        if output_id not in all_output_ids:
            raise ValueError(f"Output {output_id} not found in job outputs")

        job.selected_output_id = output_id
        self.db.commit()
        self.db.refresh(job)

        self.audit.log(
            action="generation_job.select_output",
            entity_type="generation_job",
            entity_id=job_id,
            details={"selected_output_id": output_id},
        )

        return job

    def delete_unselected_outputs(self, job_id: str) -> int:
        """Delete all unselected outputs from a completed job."""
        job = self.get_job(job_id)
        if not job or not job.selected_output_id:
            return 0

        deleted_count = 0
        output_ids = json.loads(job.output_pc_ids_json) if job.output_pc_ids_json else []
        stl_ids = json.loads(job.output_stl_ids_json) if job.output_stl_ids_json else []

        for pc_id in output_ids + stl_ids:
            if pc_id != job.selected_output_id:
                pc = self.db.query(PointCloud).filter(PointCloud.id == pc_id).first()
                if pc:
                    # Optionally delete file from disk
                    try:
                        Path(pc.file_path).unlink(missing_ok=True)
                    except Exception as e:
                        logger.warning(f"Failed to delete file {pc.file_path}: {e}")
                    self.db.delete(pc)
                    deleted_count += 1

        # Update job's output lists
        if job.selected_output_id in output_ids:
            job.output_pc_ids_json = json.dumps([job.selected_output_id])
            job.output_stl_ids_json = json.dumps([])
        else:
            job.output_pc_ids_json = json.dumps([])
            job.output_stl_ids_json = json.dumps([job.selected_output_id])

        self.db.commit()

        self.audit.log(
            action="generation_job.delete_unselected",
            entity_type="generation_job",
            entity_id=job_id,
            details={"deleted_count": deleted_count},
        )

        return deleted_count

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_generation(
        self,
        job_id: str,
        pcdiff_model_path: str | None = None,
        vox_model_path: str | None = None,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> GenerationJob:
        """
        Execute the full generation pipeline for a job.

        This method is designed to be called from a background task.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Use default model paths if not provided
        if not pcdiff_model_path:
            pcdiff_model_path = str(settings.project_root / "pcdiff" / "checkpoints" / "pcdiff_model_best.pth")
        if not vox_model_path:
            vox_model_path = str(settings.project_root / "voxelization" / "checkpoints" / "model_best.pt")

        t0 = time.time()

        try:
            # Mark as running
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            job.current_step = "Initializing"
            self.db.commit()

            self._create_notification(
                type="generation_started",
                title="Generation Started",
                message=f"Implant generation '{job.name}' has started.",
                entity_type="generation_job",
                entity_id=job.id,
            )

            def update_progress(percent: int, step: str):
                job.progress_percent = percent
                job.current_step = step
                self.db.commit()
                if progress_callback:
                    progress_callback(percent, step)

            def pcdiff_progress_callback(progress_frac: float, step_text: str, step_info: dict | None):
                """Handle PCDiff progress with detailed step info."""
                # Map PCDiff progress (0-1) to overall progress (10-60%)
                overall_percent = 10 + int(progress_frac * 50)

                # Format step text with timing info if available
                if step_info and "current_step" in step_info:
                    current = step_info["current_step"]
                    total = step_info["total_steps"]
                    step_time_ms = step_info.get("step_time_ms", 0)

                    if step_time_ms > 0:
                        # Calculate ETA
                        remaining_steps = total - current
                        eta_seconds = (remaining_steps * step_time_ms) / 1000
                        if eta_seconds > 60:
                            eta_str = f"~{int(eta_seconds / 60)}m {int(eta_seconds % 60)}s"
                        else:
                            eta_str = f"~{int(eta_seconds)}s"

                        formatted_step = f"Step {current}/{total} ({step_time_ms}ms/step, ETA: {eta_str})"
                    else:
                        formatted_step = f"Step {current}/{total}"
                else:
                    formatted_step = step_text

                update_progress(overall_percent, formatted_step)

            update_progress(5, "Loading input point cloud")

            # Load input point cloud
            input_pc = self.db.query(PointCloud).filter(PointCloud.id == job.input_pc_id).first()
            if not input_pc:
                raise ValueError(f"Input point cloud not found: {job.input_pc_id}")

            input_path = Path(input_pc.file_path)
            if not input_path.exists():
                raise ValueError(f"Input file not found: {input_pc.file_path}")

            # Load points
            input_points = np.load(str(input_path))
            if input_points.ndim == 3:
                input_points = input_points[0]
            input_points = input_points.astype(np.float32)

            update_progress(10, "Loading PCDiff model")

            # Run PCDiff inference
            implant_points, defective_points, shift, scale = self._run_pcdiff(
                input_points=input_points,
                model_path=pcdiff_model_path,
                sampling_method=job.sampling_method,
                sampling_steps=job.sampling_steps,
                num_ensemble=job.num_ensemble,
                progress_callback=pcdiff_progress_callback,
            )

            update_progress(60, "Saving ensemble point clouds")

            # Save ensemble point clouds
            output_dir = input_path.parent
            output_pc_ids = []

            for i in range(job.num_ensemble):
                implant_pc = implant_points[i]  # (3072, 3)

                # Save NPY file
                output_name = f"{job.id}_implant_ens{i}.npy"
                output_path = output_dir / output_name
                np.save(str(output_path), implant_pc)

                # Register as PointCloud
                pc = PointCloud(
                    name=f"{job.name} - Implant #{i+1}",
                    description=f"Generated implant (ensemble {i+1}/{job.num_ensemble}) from {input_pc.name}",
                    file_path=str(output_path),
                    file_format="npy",
                    file_size_bytes=get_file_size(output_path),
                    num_points=len(implant_pc),
                    point_dims=3,
                    scan_category="generated_implant",
                    defect_type=input_pc.defect_type,
                    skull_id=input_pc.skull_id,
                    project_id=job.project_id,
                    metadata_json=json.dumps({
                        "generation_job_id": job.id,
                        "ensemble_index": i,
                        "sampling_method": job.sampling_method,
                        "sampling_steps": job.sampling_steps,
                        "shift": shift.tolist(),
                        "scale": float(scale),
                    }),
                )
                self.db.add(pc)
                self.db.commit()
                self.db.refresh(pc)
                output_pc_ids.append(pc.id)

                self.audit.log(
                    action="point_cloud.generate",
                    entity_type="point_cloud",
                    entity_id=pc.id,
                    details={"generation_job_id": job.id, "ensemble_index": i},
                )

            # Update job with output IDs
            job.output_pc_ids_json = json.dumps(output_pc_ids)

            # Run voxelization for each ensemble member to create STL meshes
            update_progress(65, "Running voxelization")
            output_stl_ids = []

            # Check if voxelization model exists
            vox_model_exists = Path(vox_model_path).exists()
            if not vox_model_exists:
                logger.warning(
                    "Voxelization model not found at %s - skipping mesh generation",
                    vox_model_path
                )

            try:
                if not vox_model_exists:
                    raise FileNotFoundError(f"Voxelization model not found: {vox_model_path}")
                for i in range(job.num_ensemble):
                    # Calculate base progress for this ensemble member (65-85% range)
                    base_progress = 65 + int((i / job.num_ensemble) * 20)
                    ensemble_idx = i + 1  # Capture value for closure
                    total_ensemble = job.num_ensemble

                    # Create voxelization progress callback (use default args to capture values)
                    def vox_progress_callback(
                        step_desc: str,
                        _base=base_progress,
                        _idx=ensemble_idx,
                        _total=total_ensemble
                    ):
                        update_progress(_base, f"Voxelizing {_idx}/{_total}: {step_desc}")

                    update_progress(base_progress, f"Voxelizing implant {ensemble_idx}/{total_ensemble}")

                    # Get normalized implant points (before denormalization)
                    # We need to re-compute this from the world-space points
                    implant_world = np.load(str(output_dir / f"{job.id}_implant_ens{i}.npy"))
                    implant_normalized = (implant_world - shift) / scale

                    # Run voxelization
                    stl_path = output_dir / f"{job.id}_implant_ens{i}.stl"
                    self._run_voxelization(
                        defective_points=defective_points,
                        implant_points_normalized=implant_normalized,
                        shift=shift,
                        scale=scale,
                        output_stl_path=str(stl_path),
                        vox_model_path=vox_model_path,
                        progress_callback=vox_progress_callback,
                    )

                    # Register as PointCloud record (STL mesh)
                    stl_pc = PointCloud(
                        name=f"{job.name} - Implant #{i+1} (STL)",
                        description=f"Generated implant mesh (ensemble {i+1}/{job.num_ensemble})",
                        file_path=str(stl_path),
                        file_format="stl",
                        file_size_bytes=get_file_size(stl_path),
                        num_points=0,  # Mesh, not point cloud
                        point_dims=3,
                        scan_category="generated_implant_mesh",
                        defect_type=input_pc.defect_type,
                        skull_id=input_pc.skull_id,
                        project_id=job.project_id,
                        metadata_json=json.dumps({
                            "generation_job_id": job.id,
                            "ensemble_index": i,
                            "source_pc_id": output_pc_ids[i],
                        }),
                    )
                    self.db.add(stl_pc)
                    self.db.commit()
                    self.db.refresh(stl_pc)
                    output_stl_ids.append(stl_pc.id)

                job.output_stl_ids_json = json.dumps(output_stl_ids)
                logger.info("Voxelization completed: %d STL meshes generated", len(output_stl_ids))

            except Exception as e:
                logger.warning("Voxelization failed (point clouds still available): %s", str(e))
                # Continue without STL - point clouds are still useful

            update_progress(90, "Finalizing")

            # Mark as completed
            elapsed_ms = int((time.time() - t0) * 1000)
            job.status = "completed"
            job.progress_percent = 100
            job.current_step = "Completed"
            job.completed_at = datetime.now(timezone.utc)
            job.generation_time_ms = elapsed_ms

            # Auto-select first output if only one
            if len(output_pc_ids) == 1:
                job.selected_output_id = output_pc_ids[0]

            self.db.commit()
            self.db.refresh(job)

            self._create_notification(
                type="generation_completed",
                title="Generation Complete",
                message=f"Implant generation '{job.name}' completed successfully in {elapsed_ms/1000:.1f}s.",
                entity_type="generation_job",
                entity_id=job.id,
            )

            self.audit.log(
                action="generation_job.complete",
                entity_type="generation_job",
                entity_id=job.id,
                details={
                    "generation_time_ms": elapsed_ms,
                    "num_outputs": len(output_pc_ids),
                },
            )

            logger.info(
                "Generation job completed: %s (%d outputs, %dms)",
                job.id, len(output_pc_ids), elapsed_ms,
            )

        except Exception as e:
            logger.exception("Generation job failed: %s", job.id)
            elapsed_ms = int((time.time() - t0) * 1000)

            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            job.generation_time_ms = elapsed_ms
            self.db.commit()
            self.db.refresh(job)

            self._create_notification(
                type="generation_failed",
                title="Generation Failed",
                message=f"Implant generation '{job.name}' failed: {str(e)[:100]}",
                entity_type="generation_job",
                entity_id=job.id,
            )

            self.audit.log(
                action="generation_job.fail",
                entity_type="generation_job",
                entity_id=job.id,
                details={"error": str(e)},
            )

        return job

    # ------------------------------------------------------------------
    # PCDiff Inference
    # ------------------------------------------------------------------

    def _run_pcdiff(
        self,
        input_points: np.ndarray,
        model_path: str,
        sampling_method: str,
        sampling_steps: int,
        num_ensemble: int,
        progress_callback: Callable[[float, str, dict | None], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Run PCDiff inference to generate implant point clouds.

        Args:
            progress_callback: Callable(progress_fraction, step_text, step_info_dict)
                step_info_dict contains: current_step, total_steps, step_time_ms

        Returns:
            tuple of (implant_points, defective_points, shift, scale)
            - implant_points: (num_ensemble, 3072, 3) world-space coordinates
            - defective_points: (N, 3) original defective skull points
            - shift: (3,) center offset used for normalization
            - scale: float scale factor used for normalization
        """
        # Model hyperparameters (matching training)
        num_points = 30720
        num_nn = 3072  # Number of implant points
        sv_points = num_points - num_nn  # 27648 defective skull points

        # Ensure we have enough points
        if input_points.shape[0] < sv_points:
            # Upsample with replacement if needed
            idx = np.random.choice(input_points.shape[0], sv_points, replace=True)
        else:
            idx = np.random.choice(input_points.shape[0], sv_points, replace=False)
        partial_points_raw = input_points[idx]

        # Compute normalization from bounding box
        pc_min = partial_points_raw.min(axis=0)
        pc_max = partial_points_raw.max(axis=0)
        shift = (pc_min + pc_max) / 2.0
        scale = (pc_max - pc_min).max() / 2.0
        if scale <= 0:
            raise ValueError("Invalid scale computed from input point cloud bounding box.")
        scale = scale / 3.0  # Match training normalization

        partial_points = (partial_points_raw - shift) / scale

        if progress_callback:
            progress_callback(0.1, "Loading PCDiff model", None)

        # Load model (with caching)
        model, device = self._load_pcdiff_model(model_path, sampling_method, sampling_steps)

        if progress_callback:
            progress_callback(0.2, f"Starting {sampling_method.upper()} sampling", None)

        # Prepare input tensor
        pc_input = torch.from_numpy(partial_points).float().unsqueeze(0)  # (1, sv_points, 3)
        pc_input = pc_input.transpose(1, 2).to(device)  # (1, 3, sv_points)
        pc_input = pc_input.repeat(num_ensemble, 1, 1)  # (num_ensemble, 3, sv_points)

        noise_shape = torch.Size([num_ensemble, 3, num_nn])

        # Create diffusion progress callback
        def diffusion_progress(current_step, total_steps, step_time_ms):
            if progress_callback:
                # Map diffusion progress (0-100%) to overall progress (20-90%)
                diffusion_pct = current_step / total_steps
                overall_pct = 0.2 + diffusion_pct * 0.7
                step_info = {
                    "current_step": current_step,
                    "total_steps": total_steps,
                    "step_time_ms": step_time_ms,
                }
                progress_callback(
                    overall_pct,
                    f"Diffusion step {current_step}/{total_steps}",
                    step_info,
                )

        # Run generation
        with torch.no_grad():
            sample = model.gen_samples(
                pc_input, noise_shape, device,
                clip_denoised=False,
                sampling_method=sampling_method,
                sampling_steps=sampling_steps,
                progress_callback=diffusion_progress,
            )
            sample = sample.detach().cpu().numpy()

        if progress_callback:
            progress_callback(0.9, "Processing results", None)

        # Extract implant points and denormalize
        completed_points = sample.transpose(0, 2, 1)  # (num_ensemble, num_points, 3)
        implant_normalized = completed_points[:, sv_points:, :]  # (num_ensemble, num_nn, 3)
        implant_world = implant_normalized * scale + shift

        if progress_callback:
            progress_callback(1.0, "PCDiff inference complete", None)

        return implant_world, input_points, shift, scale

    def _load_pcdiff_model(
        self,
        model_path: str,
        sampling_method: str,
        sampling_steps: int,
    ):
        """Load PCDiff model with caching."""
        with _model_cache["lock"]:
            # Check cache
            cached = _model_cache.get("pcdiff")
            if cached and cached.get("path") == model_path:
                logger.info("Using cached PCDiff model")
                return cached["model"], cached["device"]

            logger.info("Loading PCDiff model from: %s", model_path)

            # Add project paths
            project_root = settings.project_root
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(project_root / "pcdiff"))

            from pcdiff.test_completion import Model

            # Model arguments
            class ModelArgs:
                def __init__(self):
                    self.nc = 3
                    self.num_points = 30720
                    self.num_nn = 3072
                    self.attention = True
                    self.dropout = 0.1
                    self.embed_dim = 64
                    self.sampling_method = sampling_method
                    self.sampling_steps = sampling_steps

            model_args = ModelArgs()

            # Diffusion betas
            betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float64)
            model_args.time_num = len(betas)

            # Select device based on settings
            from web_viewer.backend.services.settings_service import SettingsService
            settings_service = SettingsService(self.db)
            device_name = settings_service.get_inference_device()
            device = torch.device(device_name)
            logger.info(f"Using device: {device_name}")

            # Create model
            model = Model(
                model_args, betas, "mse", "eps", "fixedsmall",
                width_mult=1.0, vox_res_mult=1.0,
            )
            model = model.to(device)
            model.eval()

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint["model_state"]

            # Handle DDP and torch.compile checkpoints
            first_key = list(state_dict.keys())[0]
            if first_key.startswith("model.module."):
                state_dict = {k.replace("model.module.", "model."): v for k, v in state_dict.items()}
            elif first_key.startswith("model._orig_mod."):
                state_dict = {k.replace("model._orig_mod.", "model."): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            logger.info("PCDiff model loaded successfully")

            # Cache for reuse
            _model_cache["pcdiff"] = {
                "model": model,
                "device": device,
                "path": model_path,
            }

            return model, device

    # ------------------------------------------------------------------
    # Voxelization
    # ------------------------------------------------------------------

    def _run_voxelization(
        self,
        defective_points: np.ndarray,
        implant_points_normalized: np.ndarray,
        shift: np.ndarray,
        scale: float,
        output_stl_path: str,
        vox_model_path: str,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Run voxelization to convert point cloud to STL mesh.

        The voxelization model uses DPSR (Differentiable Poisson Surface Reconstruction)
        to create a watertight mesh from the combined defective skull + implant points.

        Args:
            progress_callback: Optional callable(step_description) for progress reporting
        """
        import trimesh

        if progress_callback:
            progress_callback("Preparing point cloud data")

        # Normalize defective points to [0, 1] range (same as implant)
        defective_normalized = (defective_points - shift) / scale

        # Combine defective skull + implant
        combined_points = np.concatenate(
            [defective_normalized, implant_points_normalized], axis=0
        ).astype(np.float32)

        logger.info(
            "Voxelization input: %d defective + %d implant = %d total points",
            len(defective_normalized), len(implant_points_normalized), len(combined_points)
        )

        if progress_callback:
            progress_callback("Loading voxelization model")

        # Load model and run inference
        generator, device = self._load_voxelization_model(vox_model_path)

        inputs = torch.from_numpy(combined_points).float().unsqueeze(0).to(device)

        # Create mesh generation progress callback
        def mesh_progress(step_name, step_num, total_steps):
            if progress_callback:
                progress_callback(f"{step_name} ({step_num}/{total_steps})")

        with torch.no_grad():
            vertices, faces, points, normals, psr_grid = generator.generate_mesh(
                inputs, progress_callback=mesh_progress
            )

        if progress_callback:
            progress_callback("Exporting STL mesh")

        # Scale vertices back to original coordinate space
        vertices_world = vertices * scale + shift

        # Create mesh using trimesh
        mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces)

        # Export as STL
        mesh.export(output_stl_path)
        logger.info("Saved STL mesh: %s (%d vertices, %d faces)",
                    output_stl_path, len(vertices), len(faces))

    def _load_voxelization_model(self, model_path: str):
        """Load voxelization model with caching."""
        with _model_cache["lock"]:
            cached = _model_cache.get("voxelization")
            if cached and cached.get("path") == model_path:
                logger.info("Using cached voxelization model")
                return cached["generator"], cached["device"]

            logger.info("Loading voxelization model from: %s", model_path)

            # Add project paths
            project_root = settings.project_root
            sys.path.insert(0, str(project_root))
            sys.path.insert(0, str(project_root / "voxelization"))

            from voxelization.src.model import Encode2Points
            from voxelization.src.utils import load_config, load_model_manual
            from voxelization.src import config as vox_config

            # Load config
            config_path = str(project_root / "voxelization" / "configs" / "gen_skullbreak.yaml")
            default_config = str(project_root / "voxelization" / "configs" / "default.yaml")
            cfg = load_config(config_path, default_config)

            # Override model path
            cfg['test']['model_file'] = model_path

            # Select device
            from web_viewer.backend.services.settings_service import SettingsService
            settings_service = SettingsService(self.db)
            device_name = settings_service.get_inference_device()
            device = torch.device(device_name)
            logger.info(f"Using device for voxelization: {device_name}")

            # Create and load model
            model = Encode2Points(cfg).to(device)
            state_dict = torch.load(model_path, map_location=device)
            load_model_manual(state_dict['state_dict'], model)
            model.eval()

            # Get generator
            generator = vox_config.get_generator(model, cfg, device=device)

            logger.info("Voxelization model loaded successfully")

            # Cache for reuse
            _model_cache["voxelization"] = {
                "generator": generator,
                "device": device,
                "path": model_path,
            }

            return generator, device

    # ------------------------------------------------------------------
    # Notifications
    # ------------------------------------------------------------------

    def _create_notification(
        self,
        type: str,
        title: str,
        message: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ):
        """Create a notification record."""
        notification = Notification(
            type=type,
            title=title,
            message=message,
            entity_type=entity_type,
            entity_id=entity_id,
            read=False,
        )
        self.db.add(notification)
        self.db.commit()
