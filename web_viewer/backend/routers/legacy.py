"""Legacy API endpoints (/api/*) for backward compatibility.

These preserve the original web_viewer API so the existing frontend
continues to work during migration to the v1 API.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web_viewer.backend.config import settings

router = APIRouter(tags=["legacy"])

# In-memory conversion job tracker (preserved from original main.py)
conversion_jobs: Dict[str, dict] = {}


# --- Pydantic models (unchanged from original) ---


class InferenceResult(BaseModel):
    id: str
    name: str
    has_input: bool
    has_sample: bool
    num_samples: int
    converted: bool
    web_dir: Optional[str] = None
    files: List[str] = []


class ConversionRequest(BaseModel):
    force: bool = False
    export_stl: bool = True


# --- Helpers ---


def scan_inference_results() -> List[InferenceResult]:
    results = []
    syn_dir = settings.syn_dir
    if not syn_dir.exists():
        return results

    for result_dir in syn_dir.iterdir():
        if not result_dir.is_dir():
            continue

        input_file = result_dir / "input.npy"
        sample_file = result_dir / "sample.npy"
        web_dir = result_dir / "web"

        if not input_file.exists():
            continue

        num_samples = 1
        if sample_file.exists():
            import numpy as np

            sample_data = np.load(sample_file)
            if len(sample_data.shape) == 3 and sample_data.shape[0] > 1:
                num_samples = sample_data.shape[0]

        converted = False
        files = []
        if web_dir.exists():
            files = [f.name for f in web_dir.iterdir() if f.suffix in [".ply", ".stl"]]
            converted = len(files) > 0

        results.append(
            InferenceResult(
                id=result_dir.name,
                name=result_dir.name,
                has_input=input_file.exists(),
                has_sample=sample_file.exists(),
                num_samples=num_samples,
                converted=converted,
                web_dir=str(web_dir) if web_dir.exists() else None,
                files=sorted(files),
            )
        )

    return sorted(results, key=lambda x: x.name)


async def convert_result_background(
    result_id: str, force: bool = False, export_stl: bool = True
):
    from pcdiff.utils.convert_to_web import convert_inference_result

    conversion_jobs[result_id] = {
        "status": "running",
        "message": "Converting...",
        "started_at": datetime.now(),
        "completed_at": None,
    }
    try:
        result_dir = settings.syn_dir / result_id
        if not result_dir.exists():
            raise FileNotFoundError(f"Result directory not found: {result_id}")
        stats = convert_inference_result(result_dir, force=force, export_stl=export_stl)
        conversion_jobs[result_id] = {
            "status": "completed",
            "message": f"Converted {len(stats['converted'])} files",
            "started_at": conversion_jobs[result_id]["started_at"],
            "completed_at": datetime.now(),
            "stats": stats,
        }
    except Exception as e:
        conversion_jobs[result_id] = {
            "status": "failed",
            "message": str(e),
            "started_at": conversion_jobs[result_id]["started_at"],
            "completed_at": datetime.now(),
        }


# --- Endpoints (unchanged paths) ---


@router.get("/api/status")
async def get_status():
    return {
        "status": "ok",
        "version": settings.software_version,
        "inference_results_dir": str(settings.inference_results_dir),
        "auto_convert": settings.auto_convert,
    }


@router.get("/api/results", response_model=List[InferenceResult])
async def list_results():
    try:
        return scan_inference_results()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/results/{result_id}", response_model=InferenceResult)
async def get_result(result_id: str):
    results = scan_inference_results()
    for result in results:
        if result.id == result_id:
            return result
    raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")


@router.get("/api/results/{result_id}/files")
async def list_result_files(result_id: str):
    result_dir = settings.syn_dir / result_id
    web_dir = result_dir / "web"
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
    if not web_dir.exists():
        return {"files": [], "converted": False}

    files = []
    for file in web_dir.iterdir():
        if file.suffix in [".ply", ".stl"]:
            files.append(
                {
                    "name": file.name,
                    "type": file.suffix[1:],
                    "size": file.stat().st_size,
                    "path": f"/api/files/{result_id}/{file.name}",
                }
            )
    return {"files": sorted(files, key=lambda x: x["name"]), "converted": len(files) > 0}


@router.post("/api/convert/{result_id}")
async def convert_result(
    result_id: str, request: ConversionRequest, background_tasks: BackgroundTasks
):
    result_dir = settings.syn_dir / result_id
    if not result_dir.exists():
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
    if result_id in conversion_jobs and conversion_jobs[result_id]["status"] == "running":
        return {"message": "Conversion already in progress", "status": conversion_jobs[result_id]}
    background_tasks.add_task(
        convert_result_background, result_id, request.force, request.export_stl
    )
    return {"message": "Conversion started", "result_id": result_id}


@router.post("/api/convert/batch")
async def convert_batch(request: ConversionRequest, background_tasks: BackgroundTasks):
    from pcdiff.utils.convert_to_web import batch_convert_all

    async def batch_convert_background():
        conversion_jobs["_batch"] = {
            "status": "running",
            "message": "Batch converting...",
            "started_at": datetime.now(),
            "completed_at": None,
        }
        try:
            stats = batch_convert_all(
                settings.inference_results_dir,
                force=request.force,
                export_stl=request.export_stl,
            )
            conversion_jobs["_batch"] = {
                "status": "completed",
                "message": f"Converted {stats['total']} results",
                "started_at": conversion_jobs["_batch"]["started_at"],
                "completed_at": datetime.now(),
                "stats": stats,
            }
        except Exception as e:
            conversion_jobs["_batch"] = {
                "status": "failed",
                "message": str(e),
                "started_at": conversion_jobs["_batch"]["started_at"],
                "completed_at": datetime.now(),
            }

    background_tasks.add_task(batch_convert_background)
    return {"message": "Batch conversion started", "check_status_at": "/api/conversion-status/_batch"}


@router.get("/api/conversion-status/{result_id}")
async def get_conversion_status(result_id: str):
    if result_id not in conversion_jobs:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    return conversion_jobs[result_id]


@router.get("/api/files/{result_id}/{filename}")
async def serve_file(result_id: str, filename: str):
    from pcdiff.utils.convert_to_web import convert_inference_result

    result_dir = settings.syn_dir / result_id
    web_dir = result_dir / "web"
    file_path = web_dir / filename

    if not file_path.exists():
        if settings.auto_convert:
            try:
                convert_inference_result(result_dir, force=False, export_stl=settings.export_stl)
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found after conversion")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Conversion failed: {str(e)}")
        else:
            raise HTTPException(status_code=404, detail="File not found")

    media_type = "application/octet-stream"
    if filename.endswith(".ply"):
        media_type = "application/ply"
    elif filename.endswith(".stl"):
        media_type = "application/sla"

    return FileResponse(path=file_path, media_type=media_type, filename=filename)
