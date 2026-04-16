"""
Seed the SkullBreak dataset into the database.

Scans the directory structure and registers all NRRD volumes
and NPY point clouds. Can be run standalone or imported.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def seed_skullbreak(base_dir: str | Path | None = None) -> dict:
    """Import the SkullBreak dataset into the database.

    Args:
        base_dir: Path to SkullBreak directory. If None, uses settings.

    Returns:
        Import statistics dict.
    """
    # Ensure project root is importable
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from crainial_app.backend.config import settings
    from crainial_app.backend.database import SessionLocal
    from crainial_app.backend.services.audit_service import AuditService
    from crainial_app.backend.services.scan_service import ScanService

    if base_dir is None:
        base_dir = settings.skullbreak_dir

    db = SessionLocal()
    try:
        audit = AuditService(db)
        scan_svc = ScanService(db, audit)
        stats = scan_svc.import_skullbreak(base_dir)

        logger.info(
            "SkullBreak import complete: %d scans, %d point clouds, %d skipped, %d errors",
            stats["scans_created"],
            stats["point_clouds_created"],
            stats["skipped"],
            len(stats["errors"]),
        )
        return stats
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Seed SkullBreak dataset")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Path to SkullBreak directory (default: from settings)",
    )
    args = parser.parse_args()
    seed_skullbreak(args.base_dir)
