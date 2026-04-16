"""
Configuration for PCDiff Web Viewer Backend

Settings are loaded from environment variables with the PCDIFF_ prefix.
For regulatory compliance (IEC 62304, ISO 13485), all configuration
changes must be traceable via the audit_log.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8081

    # Directory paths
    project_root: Path = Path(__file__).parent.parent.parent
    inference_results_dir: Path = project_root / "inference_results_ddim50"
    data_dir: Path = project_root / "datasets"
    skullbreak_dir: Path = project_root / "datasets" / "SkullBreak"

    # Database
    database_url: str = ""
    sql_echo: bool = False

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
    export_stl: bool = True  # Export STL files alongside PLY

    # Audit / regulatory
    software_version: str = "2.0.0"

    # Volume serving
    max_volume_cache_mb: int = 512
    nrrd_downsample_threshold: int = 256

    class Config:
        env_prefix = "PCDIFF_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add external origin if provided
        if self.external_origin:
            self.cors_origins.append(self.external_origin)
        # Default database URL next to the backend code
        if not self.database_url:
            db_dir = self.project_root / "crainial_app" / "data"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.database_url = f"sqlite:///{db_dir / 'pcdiff_viewer.db'}"

    @property
    def syn_dir(self) -> Path:
        """Get the synthetic results directory."""
        return self.inference_results_dir / "syn"


# Global settings instance
settings = Settings()
