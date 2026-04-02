"""
API endpoints for application settings.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.settings import (
    AllSettingsRead,
    AllSettingsUpdate,
    SettingRead,
    SettingUpdate,
    SystemInfoRead,
)
from web_viewer.backend.services.settings_service import SettingsService

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def get_settings_service(db: Session = Depends(get_db)) -> SettingsService:
    return SettingsService(db)


@router.get("/", response_model=AllSettingsRead)
def get_all_settings(
    service: SettingsService = Depends(get_settings_service),
):
    """Get all application settings."""
    settings_dict = service.get_all_as_dict()

    # Check if API key is set (don't expose the actual key)
    runpod_api_key = settings_dict.get("runpod_api_key", "")
    runpod_api_key_set = bool(runpod_api_key and len(runpod_api_key) > 0)

    return AllSettingsRead(
        inference_device=settings_dict.get("inference_device", "auto"),
        default_sampling_method=settings_dict.get("default_sampling_method", "ddim"),
        default_sampling_steps=int(settings_dict.get("default_sampling_steps", "50")),
        default_ensemble_count=int(settings_dict.get("default_ensemble_count", "5")),
        pcdiff_model_path=settings_dict.get("pcdiff_model_path", ""),
        voxelization_model_path=settings_dict.get("voxelization_model_path", ""),
        # Cloud settings
        cloud_generation_enabled=settings_dict.get("cloud_generation_enabled", "false").lower() == "true",
        runpod_endpoint_id=settings_dict.get("runpod_endpoint_id", ""),
        runpod_api_key_set=runpod_api_key_set,
        aws_s3_bucket=settings_dict.get("aws_s3_bucket", ""),
        aws_s3_region=settings_dict.get("aws_s3_region", "us-east-1"),
    )


@router.put("/", response_model=AllSettingsRead)
def update_settings(
    body: AllSettingsUpdate,
    service: SettingsService = Depends(get_settings_service),
):
    """Update multiple settings at once."""
    updates = {}
    if body.inference_device is not None:
        updates["inference_device"] = body.inference_device
    if body.default_sampling_method is not None:
        updates["default_sampling_method"] = body.default_sampling_method
    if body.default_sampling_steps is not None:
        updates["default_sampling_steps"] = str(body.default_sampling_steps)
    if body.default_ensemble_count is not None:
        updates["default_ensemble_count"] = str(body.default_ensemble_count)
    if body.pcdiff_model_path is not None:
        updates["pcdiff_model_path"] = body.pcdiff_model_path
    if body.voxelization_model_path is not None:
        updates["voxelization_model_path"] = body.voxelization_model_path
    # Cloud settings
    if body.cloud_generation_enabled is not None:
        updates["cloud_generation_enabled"] = str(body.cloud_generation_enabled).lower()
    if body.runpod_endpoint_id is not None:
        updates["runpod_endpoint_id"] = body.runpod_endpoint_id
    if body.runpod_api_key is not None:
        updates["runpod_api_key"] = body.runpod_api_key
    if body.aws_s3_bucket is not None:
        updates["aws_s3_bucket"] = body.aws_s3_bucket
    if body.aws_s3_region is not None:
        updates["aws_s3_region"] = body.aws_s3_region

    service.update_multiple(updates)
    settings_dict = service.get_all_as_dict()

    # Check if API key is set (don't expose the actual key)
    runpod_api_key = settings_dict.get("runpod_api_key", "")
    runpod_api_key_set = bool(runpod_api_key and len(runpod_api_key) > 0)

    return AllSettingsRead(
        inference_device=settings_dict.get("inference_device", "auto"),
        default_sampling_method=settings_dict.get("default_sampling_method", "ddim"),
        default_sampling_steps=int(settings_dict.get("default_sampling_steps", "50")),
        default_ensemble_count=int(settings_dict.get("default_ensemble_count", "5")),
        pcdiff_model_path=settings_dict.get("pcdiff_model_path", ""),
        voxelization_model_path=settings_dict.get("voxelization_model_path", ""),
        # Cloud settings
        cloud_generation_enabled=settings_dict.get("cloud_generation_enabled", "false").lower() == "true",
        runpod_endpoint_id=settings_dict.get("runpod_endpoint_id", ""),
        runpod_api_key_set=runpod_api_key_set,
        aws_s3_bucket=settings_dict.get("aws_s3_bucket", ""),
        aws_s3_region=settings_dict.get("aws_s3_region", "us-east-1"),
    )


@router.get("/system-info", response_model=SystemInfoRead)
def get_system_info(
    service: SettingsService = Depends(get_settings_service),
):
    """Get system information for device selection."""
    info = SettingsService.get_system_info()

    # Add cloud configuration status
    settings_dict = service.get_all_as_dict()
    runpod_endpoint_id = settings_dict.get("runpod_endpoint_id", "")
    runpod_api_key = settings_dict.get("runpod_api_key", "")

    info["cloud_configured"] = bool(runpod_endpoint_id and runpod_api_key)
    info["runpod_endpoint_id"] = runpod_endpoint_id if runpod_endpoint_id else None

    return info


@router.get("/{key}", response_model=SettingRead)
def get_setting(
    key: str,
    service: SettingsService = Depends(get_settings_service),
):
    """Get a specific setting by key."""
    setting = service.get(key)
    if not setting:
        raise HTTPException(status_code=404, detail=f"Setting '{key}' not found")
    return setting


@router.put("/{key}", response_model=SettingRead)
def update_setting(
    key: str,
    body: SettingUpdate,
    service: SettingsService = Depends(get_settings_service),
):
    """Update a specific setting."""
    setting = service.set(key, body.value)
    return setting


@router.post("/init-defaults", status_code=204)
def init_default_settings(
    service: SettingsService = Depends(get_settings_service),
):
    """Initialize default settings (for setup/reset)."""
    service.init_defaults()
