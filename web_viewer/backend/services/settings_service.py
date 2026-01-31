"""
Service for managing application settings.
"""
import json
import logging
import platform
import sys

import torch
from sqlalchemy.orm import Session

from web_viewer.backend.models.settings import Setting, DEFAULT_SETTINGS

logger = logging.getLogger(__name__)


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
        """Get just the value of a setting."""
        setting = self.get(key)
        if setting:
            return setting.value
        return default

    def get_all(self) -> list[Setting]:
        """Get all settings."""
        return self.db.query(Setting).all()

    def get_all_as_dict(self) -> dict[str, str]:
        """Get all settings as a dictionary."""
        settings = self.get_all()
        result = {}
        for s in settings:
            result[s.key] = s.value
        # Fill in any missing defaults
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
        """Get system information for the settings page."""
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

        device_name = None
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
        elif mps_available:
            device_name = "Apple Silicon (MPS)"

        # Try to get the PVCNN backend type
        backend_type = "unknown"
        try:
            import sys
            from pathlib import Path
            # Add pcdiff to path for import
            project_root = Path(__file__).parent.parent.parent.parent
            pcdiff_path = str(project_root / "pcdiff")
            if pcdiff_path not in sys.path:
                sys.path.insert(0, pcdiff_path)
            from modules.functional.backend import get_backend_type
            backend_type = get_backend_type()
        except Exception as e:
            logger.debug(f"Could not determine backend type: {e}")
            backend_type = "cpu"  # Default to CPU on non-CUDA systems

        return {
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "device_name": device_name,
            "torch_version": torch.__version__,
            "backend_type": backend_type,
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        }

    def get_inference_device(self) -> str:
        """
        Get the configured inference device, resolving 'auto' to the best available.

        Returns: 'cuda', 'mps', or 'cpu'
        """
        device_setting = self.get_value("inference_device", "auto")

        if device_setting == "auto":
            # Auto-detect best available device
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        elif device_setting == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        elif device_setting == "mps":
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            logger.warning("MPS requested but not available, falling back to CPU")
            return "cpu"
        else:
            return "cpu"
