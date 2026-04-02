"""
FastAPI Backend for PCDiff Web Viewer

Provides REST API endpoints for:
- Managing projects, scans (NRRD/DICOM), and point clouds (NPY)
- Serving binary data for vtk.js and Three.js 3D viewers
- SDF computation for point cloud colorization
- Color profile management
- Audit trail for regulatory compliance (MDR / IEC 62304 / ISO 13485)
- Legacy endpoints for backward compatibility with v1.0 frontend
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    web_viewer_dir = Path(__file__).parent.parent
    env_file = web_viewer_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from {env_file}")
except ImportError:
    pass  # python-dotenv not installed, skip

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_viewer.backend.config import settings
from web_viewer.backend.database import init_db
from web_viewer.backend.middleware.audit_middleware import AuditMiddleware

# Import routers
from web_viewer.backend.routers import (
    audit,
    case_reports,
    color_profiles,
    filesystem,
    fit_metrics,
    generation_jobs,
    legacy,
    notifications,
    patients,
    point_clouds,
    projects,
    scans,
    viewer,
)
from web_viewer.backend.routers import (
    settings as settings_router,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info("Initialising database...")
    init_db()

    # Run seed if database is empty
    _seed_defaults_if_needed()

    logger.info("PCDiff Web Viewer API v%s ready", settings.software_version)
    logger.info("Database: %s", settings.database_url)
    logger.info("Data directory: %s", settings.data_dir)
    yield
    logger.info("Shutting down PCDiff Web Viewer API")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="PCDiff Web Viewer API",
        description="API for 3D medical data viewing, point cloud management, and cranial implant generation",
        version=settings.software_version,
        lifespan=lifespan,
    )

    # Middleware (order matters -- outermost first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-ID",
            "X-Num-Points",
            "X-Point-Dims",
            "X-Dtype",
            "X-Count",
            "X-Volume-Metadata",
            "X-Num-Faces",
            "X-Is-Watertight",
        ],
    )
    app.add_middleware(AuditMiddleware)

    # v1 API routers
    app.include_router(projects.router)
    app.include_router(scans.router)
    app.include_router(point_clouds.router)
    app.include_router(color_profiles.router)
    app.include_router(fit_metrics.router)
    app.include_router(generation_jobs.router)
    app.include_router(notifications.router)
    app.include_router(settings_router.router)
    app.include_router(viewer.router)
    app.include_router(audit.router)
    app.include_router(filesystem.router)
    # Doctor portal routers
    app.include_router(patients.router)
    app.include_router(case_reports.router)

    # Legacy endpoints (backward compatibility)
    app.include_router(legacy.router)

    return app


def _seed_defaults_if_needed():
    """Seed default color profiles and settings if the tables are empty."""
    from web_viewer.backend.database import SessionLocal
    from web_viewer.backend.models.color_profile import ColorProfile
    from web_viewer.backend.services.settings_service import SettingsService

    db = SessionLocal()
    try:
        # Seed color profiles
        count = db.query(ColorProfile).count()
        if count == 0:
            logger.info("Seeding default color profiles...")
            from web_viewer.backend.seed.seed_color_profiles import seed_color_profiles

            seed_color_profiles(db)
            logger.info("Default color profiles seeded")

        # Seed default settings
        logger.info("Initializing default settings...")
        settings_service = SettingsService(db)
        settings_service.init_defaults()
        logger.info("Default settings initialized")
    finally:
        db.close()


# Create the app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    print(f"Starting PCDiff Web Viewer API v{settings.software_version}")
    print(f"Server: http://{settings.host}:{settings.port}")
    print(f"Database: {settings.database_url}")
    print(f"CORS origins: {settings.cors_origins}")

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
