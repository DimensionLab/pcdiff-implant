"""
Application settings model.

Stores user-configurable settings for the CrAInial cran-2 application.
Uses a key-value approach for flexibility.
"""

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from crainial_app.backend.database import Base
from crainial_app.backend.models.base import AuditMixin, UUIDMixin


class Setting(UUIDMixin, AuditMixin, Base):
    """Key-value settings storage."""

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
        "description": "Device for running inference (auto, cuda, mps, cpu).",
    },
    "cran2_threshold": {
        "value": "0.5",
        "description": "Binarization threshold (0..1) applied to the cran-2 implant probability map.",
    },
    # Cloud generation settings
    "cloud_generation_enabled": {
        "value": "true",
        "description": "Enable cran-2 cloud inference via Runpod serverless GPU.",
    },
    "runpod_endpoint_id": {
        "value": "wferq1g3i1hhqd",
        "description": "Runpod serverless endpoint ID for the cran-2 model.",
    },
    "runpod_api_key": {
        "value": "",
        "description": "Runpod API key for authentication (stored securely).",
    },
    "aws_s3_bucket": {
        "value": "test-crainial",
        "description": "AWS S3 bucket name for cran-2 generated implant NRRD outputs.",
    },
    "aws_s3_region": {
        "value": "eu-central-1",
        "description": "AWS S3 region for the results bucket.",
    },
}
