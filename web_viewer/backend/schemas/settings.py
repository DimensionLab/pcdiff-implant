"""
Pydantic schemas for application settings.
"""
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
    """Schema for reading all settings as a dictionary."""

    inference_device: str = "auto"
    default_sampling_method: str = "ddim"
    default_sampling_steps: int = 50
    default_ensemble_count: int = 5
    pcdiff_model_path: str = "pcdiff/checkpoints/pcdiff_model_best.pth"
    voxelization_model_path: str = "voxelization/checkpoints/model_best.pt"


class AllSettingsUpdate(BaseModel):
    """Schema for updating multiple settings at once."""

    inference_device: Literal["auto", "cuda", "mps", "cpu"] | None = None
    default_sampling_method: Literal["ddim", "ddpm"] | None = None
    default_sampling_steps: int | None = Field(None, ge=10, le=1000)
    default_ensemble_count: int | None = Field(None, ge=1, le=10)
    pcdiff_model_path: str | None = None
    voxelization_model_path: str | None = None


class SystemInfoRead(BaseModel):
    """System information for the settings page."""

    cuda_available: bool
    mps_available: bool
    device_name: str | None
    torch_version: str
    backend_type: str  # 'cuda' or 'cpu'
    python_version: str
    platform: str
