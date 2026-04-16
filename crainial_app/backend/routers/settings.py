"""API endpoints for application settings (cran-2)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.settings import (
    AllSettingsRead,
    AllSettingsUpdate,
    SettingRead,
    SettingUpdate,
    SystemInfoRead,
)
from crainial_app.backend.services.settings_service import SettingsService

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def get_settings_service(db: Session = Depends(get_db)) -> SettingsService:
    return SettingsService(db)


def _build_all_settings(settings_dict: dict) -> AllSettingsRead:
    runpod_api_key = settings_dict.get("runpod_api_key", "")
    return AllSettingsRead(
        inference_device=settings_dict.get("inference_device", "auto"),
        cran2_threshold=float(settings_dict.get("cran2_threshold", "0.5")),
        cloud_generation_enabled=settings_dict.get("cloud_generation_enabled", "true").lower() == "true",
        runpod_endpoint_id=settings_dict.get("runpod_endpoint_id", "wferq1g3i1hhqd"),
        runpod_api_key_set=bool(runpod_api_key),
        aws_s3_bucket=settings_dict.get("aws_s3_bucket", "test-crainial"),
        aws_s3_region=settings_dict.get("aws_s3_region", "eu-central-1"),
    )


@router.get("/", response_model=AllSettingsRead)
def get_all_settings(service: SettingsService = Depends(get_settings_service)):
    """Get all application settings."""
    return _build_all_settings(service.get_all_as_dict())


@router.put("/", response_model=AllSettingsRead)
def update_settings(
    body: AllSettingsUpdate,
    service: SettingsService = Depends(get_settings_service),
):
    """Update multiple settings at once."""
    updates: dict[str, str] = {}
    if body.inference_device is not None:
        updates["inference_device"] = body.inference_device
    if body.cran2_threshold is not None:
        updates["cran2_threshold"] = str(body.cran2_threshold)
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
    return _build_all_settings(service.get_all_as_dict())


@router.get("/system-info", response_model=SystemInfoRead)
def get_system_info(service: SettingsService = Depends(get_settings_service)):
    """Get system information for the settings page."""
    info = SettingsService.get_system_info()
    settings_dict = service.get_all_as_dict()
    runpod_endpoint_id = settings_dict.get("runpod_endpoint_id", "")
    runpod_api_key = settings_dict.get("runpod_api_key", "")
    info["cloud_configured"] = bool(runpod_endpoint_id and runpod_api_key)
    info["runpod_endpoint_id"] = runpod_endpoint_id or None
    return info


@router.get("/{key}", response_model=SettingRead)
def get_setting(key: str, service: SettingsService = Depends(get_settings_service)):
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
    return service.set(key, body.value)


@router.post("/init-defaults", status_code=204)
def init_default_settings(service: SettingsService = Depends(get_settings_service)):
    """Initialize default settings (for setup/reset)."""
    service.init_defaults()
