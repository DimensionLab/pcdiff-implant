"""
Application settings model.

Stores user-configurable settings for the CrAInial application.
Uses a key-value approach for flexibility.
"""
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from web_viewer.backend.database import Base
from web_viewer.backend.models.base import AuditMixin, UUIDMixin


class Setting(UUIDMixin, AuditMixin, Base):
    """
    Key-value settings storage.

    Stores application settings as key-value pairs with JSON support for complex values.
    """

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<Setting {self.key}={self.value}>"


# Default settings with their descriptions
DEFAULT_SETTINGS = {
    "inference_device": {
        "value": "auto",
        "description": "Device for running inference (auto, cuda, mps, cpu). 'auto' will select the best available device."
    },
    "default_sampling_method": {
        "value": "ddim",
        "description": "Default sampling method for generation (ddim or ddpm)"
    },
    "default_sampling_steps": {
        "value": "50",
        "description": "Default number of sampling steps"
    },
    "default_ensemble_count": {
        "value": "5",
        "description": "Default number of ensemble samples"
    },
    "pcdiff_model_path": {
        "value": "pcdiff/checkpoints/pcdiff_model_best.pth",
        "description": "Path to PCDiff model checkpoint (relative to project root)"
    },
    "voxelization_model_path": {
        "value": "voxelization/checkpoints/model_best.pt",
        "description": "Path to voxelization model checkpoint (relative to project root)"
    },
    # Cloud generation settings
    "cloud_generation_enabled": {
        "value": "false",
        "description": "Enable cloud-based generation using Runpod serverless GPU"
    },
    "runpod_endpoint_id": {
        "value": "",
        "description": "Runpod serverless endpoint ID (e.g., '6on3tc0nzlyt42')"
    },
    "runpod_api_key": {
        "value": "",
        "description": "Runpod API key for authentication (stored securely)"
    },
    "aws_s3_bucket": {
        "value": "",
        "description": "AWS S3 bucket name for storing cloud generation results"
    },
    "aws_s3_region": {
        "value": "us-east-1",
        "description": "AWS S3 region for the results bucket"
    },
}
