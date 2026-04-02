"""
File system utilities: validation, checksums, metadata extraction.

All file access is read-only. Files are never moved, copied, or modified
by this service -- they remain in place on disk and are referenced by path.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Allowed extensions for each category
ALLOWED_SCAN_EXTENSIONS = {".nrrd"}
ALLOWED_POINT_CLOUD_EXTENSIONS = {".npy", ".ply", ".stl"}


def validate_file_path(path: str, allowed_extensions: set[str] | None = None) -> Path:
    """Validate that a file path is absolute, exists, and is readable.

    Raises:
        ValueError: If the path is invalid, does not exist, or has a
                    disallowed extension.
    """
    p = Path(path)
    if not p.is_absolute():
        raise ValueError(f"Path must be absolute: {path}")
    if not p.exists():
        raise ValueError(f"File does not exist: {path}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if allowed_extensions and p.suffix.lower() not in allowed_extensions:
        raise ValueError(f"Unsupported file extension '{p.suffix}'. Allowed: {allowed_extensions}")
    return p


def compute_sha256(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute SHA-256 checksum for regulatory provenance tracking."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_nrrd_metadata(file_path: Path) -> dict:
    """Read NRRD header without loading the full volume.

    Returns a dict with keys: dims, spacing, space_directions,
    encoding, type, and the raw header dict.
    """
    try:
        import nrrd

        header = nrrd.read_header(str(file_path))
    except Exception as e:
        logger.warning("Failed to read NRRD header from %s: %s", file_path, e)
        return {}

    result: dict = {"raw_header": {k: _serializable(v) for k, v in header.items()}}

    # Volume dimensions (convert numpy int64 to Python int for SQLite/Pydantic)
    sizes = header.get("sizes")
    if sizes is not None:
        dims = [int(d) for d in sizes]
        result["dims"] = dims

    # Voxel spacing
    spacings = header.get("spacings")
    space_dirs = header.get("space directions")
    if spacings is not None:
        result["spacing"] = [float(s) for s in spacings]
    elif space_dirs is not None:
        # Extract spacing from space directions diagonal
        spacing = []
        for row in space_dirs:
            if row is not None and hasattr(row, "__len__"):
                spacing.append(float(sum(x**2 for x in row) ** 0.5))
        if spacing:
            result["spacing"] = spacing

    result["encoding"] = header.get("encoding", "unknown")
    result["type"] = header.get("type", "unknown")

    return result


def read_npy_metadata(file_path: Path) -> dict:
    """Read NPY header to get shape and dtype without loading full array."""
    try:
        with open(file_path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        return {
            "shape": list(shape),
            "dtype": str(dtype),
            "num_points": shape[0] if len(shape) >= 1 else 0,
            "point_dims": shape[1] if len(shape) >= 2 else 0,
        }
    except Exception as e:
        logger.warning("Failed to read NPY header from %s: %s", file_path, e)
        return {}


def get_file_size(file_path: Path) -> int:
    """Get file size in bytes."""
    return file_path.stat().st_size


def _serializable(val):
    """Convert numpy types to Python builtins for JSON serialisation."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return val
