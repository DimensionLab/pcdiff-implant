"""
Service for managing application settings.

Settings can be configured via:
1. Database (through the UI settings page)
2. Environment variables (as fallback, especially for secrets)

Environment variable mapping:
- RUNPOD_API_KEY: Runpod API key for cloud generation
- RUNPOD_ENDPOINT_ID: Runpod endpoint ID
- AWS_S3_BUCKET: S3 bucket for cloud results
- AWS_S3_REGION: S3 region
"""

import logging
import os
import platform
import sys

from sqlalchemy.orm import Session

from crainial_app.backend.models.settings import DEFAULT_SETTINGS, Setting

logger = logging.getLogger(__name__)

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# Environment variable mapping for settings
ENV_VAR_MAPPING = {
    "runpod_api_key": "RUNPOD_API_KEY",
    "runpod_endpoint_id": "RUNPOD_ENDPOINT_ID",
    "aws_s3_bucket": "AWS_S3_BUCKET",
    "aws_s3_region": "AWS_S3_REGION",
}


class SettingsService:
    """Service for CRUD operations on application settings."""

    def __init__(self, db: Session):
        self.db = db

    def init_defaults(self) -> None:
        """Initialize default settings if they don't exist."""
        for key, config in DEFAULT_SETTINGS.items():
            existing = self.db.query(Setting).filter(Setting.key == key).first()
            if not existing:
                setting = Setting(
                    key=key,
                    value=config["value"],
                    description=config["description"],
                )
                self.db.add(setting)
                logger.info(f"Created default setting: {key}")
        self.db.commit()

    def get(self, key: str) -> Setting | None:
        """Get a setting by key."""
        return self.db.query(Setting).filter(Setting.key == key).first()

    def get_value(self, key: str, default: str | None = None) -> str | None:
        """
        Get just the value of a setting.

        Priority:
        1. Database setting (if set and non-empty)
        2. Environment variable (if mapped and set)
        3. Default value
        """
        setting = self.get(key)
        if setting and setting.value:
            return setting.value

        env_var = ENV_VAR_MAPPING.get(key)
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                logger.debug(f"Using environment variable {env_var} for setting {key}")
                return env_value

        return default

    def get_all(self) -> list[Setting]:
        """Get all settings."""
        return self.db.query(Setting).all()

    def get_all_as_dict(self) -> dict[str, str]:
        """Get all settings as a dictionary."""
        settings = self.get_all()
        result = {s.key: s.value for s in settings}
        for key, config in DEFAULT_SETTINGS.items():
            if key not in result:
                result[key] = config["value"]
        return result

    def set(self, key: str, value: str, description: str | None = None) -> Setting:
        """Set a setting value (creates if doesn't exist)."""
        setting = self.get(key)
        if setting:
            setting.value = value
            if description is not None:
                setting.description = description
        else:
            setting = Setting(key=key, value=value, description=description)
            self.db.add(setting)
        self.db.commit()
        self.db.refresh(setting)
        return setting

    def update_multiple(self, updates: dict[str, str]) -> dict[str, str]:
        """Update multiple settings at once."""
        for key, value in updates.items():
            if value is not None:
                self.set(key, str(value))
        return self.get_all_as_dict()

    def delete(self, key: str) -> bool:
        """Delete a setting."""
        setting = self.get(key)
        if setting:
            self.db.delete(setting)
            self.db.commit()
            return True
        return False

    @staticmethod
    def get_system_info() -> dict:
        """Return system information for the settings page.

        Inference for cran-2 happens on Runpod, so the local torch install is
        purely informational here (and may be absent in the runtime image).
        """
        cuda_available = False
        mps_available = False
        device_name: str | None = None
        torch_version = "not installed"

        if _TORCH_AVAILABLE:
            try:
                cuda_available = bool(torch.cuda.is_available())
                mps_available = bool(
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                )
                if cuda_available:
                    device_name = torch.cuda.get_device_name(0)
                elif mps_available:
                    device_name = "Apple Silicon (MPS)"
                torch_version = getattr(torch, "__version__", "unknown")
            except Exception as e:
                logger.debug(f"torch probe failed: {e}")

        backend_type = "cuda" if cuda_available else "mps" if mps_available else "cpu"

        return {
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "device_name": device_name,
            "torch_version": torch_version,
            "backend_type": backend_type,
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        }

    def get_inference_device(self) -> str:
        """Resolve the configured inference device against what's available locally.

        With cran-2 the heavy lifting is remote, so this only matters for
        small local utilities (preview rendering etc.).
        """
        device_setting = (self.get_value("inference_device", "auto") or "auto").lower()

        if not _TORCH_AVAILABLE:
            return "cpu"

        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

        if device_setting == "auto":
            if cuda_ok:
                return "cuda"
            if mps_ok:
                return "mps"
            return "cpu"
        if device_setting == "cuda":
            return "cuda" if cuda_ok else "cpu"
        if device_setting == "mps":
            return "mps" if mps_ok else "cpu"
        return "cpu"
