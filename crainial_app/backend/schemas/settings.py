"""Pydantic schemas for cran-2 application settings."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SettingRead(BaseModel):
    """Schema for reading a single setting."""

    id: str
    key: str
    value: str
    description: str | None
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class SettingUpdate(BaseModel):
    """Schema for updating a setting value."""

    value: str = Field(..., min_length=1)


class SettingCreate(BaseModel):
    """Schema for creating a new setting (admin only)."""

    key: str = Field(..., min_length=1, max_length=100)
    value: str = Field(..., min_length=1)
    description: str | None = None


class AllSettingsRead(BaseModel):
    """Schema for reading all settings as a single object."""

    inference_device: str = "auto"
    # cran-2 inference defaults
    cran2_threshold: float = 0.5
    # Cloud generation settings
    cloud_generation_enabled: bool = True
    runpod_endpoint_id: str = "wferq1g3i1hhqd"
    runpod_api_key_set: bool = False  # whether an API key is configured
    aws_s3_bucket: str = "test-crainial"
    aws_s3_region: str = "eu-central-1"


class AllSettingsUpdate(BaseModel):
    """Schema for updating multiple settings at once."""

    inference_device: Literal["auto", "cuda", "mps", "cpu"] | None = None
    cran2_threshold: float | None = Field(None, ge=0.0, le=1.0)
    cloud_generation_enabled: bool | None = None
    runpod_endpoint_id: str | None = None
    runpod_api_key: str | None = None  # write-only
    aws_s3_bucket: str | None = None
    aws_s3_region: str | None = None


class SystemInfoRead(BaseModel):
    """System information for the settings page."""

    cuda_available: bool
    mps_available: bool
    device_name: str | None
    torch_version: str
    backend_type: str  # 'cuda' or 'cpu'
    python_version: str
    platform: str
    cloud_configured: bool = False
    runpod_endpoint_id: str | None = None
