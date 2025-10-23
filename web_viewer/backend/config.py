"""
Configuration for PCDiff Web Viewer Backend
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080
    
    # Directory paths
    project_root: Path = Path(__file__).parent.parent.parent
    inference_results_dir: Path = project_root / "inference_results_ddim50"
    
    # CORS settings
    cors_origins: list = [
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",  # Alternative React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    # Allow adding external IPs via environment variable
    external_origin: str = ""
    
    # Conversion settings
    auto_convert: bool = True  # Automatically convert on access
    export_stl: bool = True    # Export STL files alongside PLY
    
    class Config:
        env_prefix = "PCDIFF_"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add external origin if provided
        if self.external_origin:
            self.cors_origins.append(self.external_origin)
    
    @property
    def syn_dir(self) -> Path:
        """Get the synthetic results directory."""
        return self.inference_results_dir / "syn"


# Global settings instance
settings = Settings()

