"""
API endpoints for filesystem browsing.
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/filesystem", tags=["filesystem"])


class FileEntry(BaseModel):
    name: str
    path: str
    is_dir: bool
    size: Optional[int] = None
    extension: Optional[str] = None


class DirectoryListing(BaseModel):
    path: str
    parent: Optional[str]
    entries: List[FileEntry]


# Allowed file extensions for data import
ALLOWED_EXTENSIONS = {".nrrd", ".npy", ".ply", ".stl", ".obj"}


@router.get("/browse", response_model=DirectoryListing)
def browse_directory(
    path: str = Query(default="~", description="Directory path to browse"),
    show_hidden: bool = Query(default=False, description="Show hidden files"),
    filter_extensions: bool = Query(default=True, description="Only show allowed data file extensions"),
):
    """
    Browse a directory and return its contents.

    Returns directories and files that can be registered as data files.
    """
    # Expand ~ to home directory
    if path.startswith("~"):
        path = os.path.expanduser(path)

    # Normalize path
    try:
        dir_path = Path(path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid path: {path}")

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path does not exist: {path}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    entries: List[FileEntry] = []

    try:
        for item in sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            # Skip hidden files unless requested
            if not show_hidden and item.name.startswith("."):
                continue

            is_dir = item.is_dir()
            extension = item.suffix.lower() if not is_dir else None

            # Filter files by extension if requested
            if not is_dir and filter_extensions:
                if extension not in ALLOWED_EXTENSIONS:
                    continue

            try:
                size = item.stat().st_size if not is_dir else None
            except (OSError, PermissionError):
                size = None

            entries.append(
                FileEntry(
                    name=item.name,
                    path=str(item),
                    is_dir=is_dir,
                    size=size,
                    extension=extension,
                )
            )
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")

    # Get parent directory
    parent = str(dir_path.parent) if dir_path.parent != dir_path else None

    return DirectoryListing(
        path=str(dir_path),
        parent=parent,
        entries=entries,
    )


@router.get("/home", response_model=dict)
def get_home_directory():
    """Get the user's home directory path."""
    return {"path": os.path.expanduser("~")}


@router.get("/common-paths", response_model=List[dict])
def get_common_paths():
    """Get common/useful paths for quick navigation."""
    home = Path.home()

    paths = [
        {"name": "Home", "path": str(home)},
        {"name": "Desktop", "path": str(home / "Desktop")},
        {"name": "Documents", "path": str(home / "Documents")},
        {"name": "Downloads", "path": str(home / "Downloads")},
    ]

    # Add project-specific paths if they exist
    project_paths = [
        {
            "name": "SkullBreak Dataset",
            "path": str(home / "projects/dimensionlab/pcdiff-implant/pcdiff/datasets/SkullBreak"),
        },
        {"name": "PCDiff Datasets", "path": str(home / "projects/dimensionlab/pcdiff-implant/pcdiff/datasets")},
    ]

    # Only include paths that exist
    result = []
    for p in paths:
        if Path(p["path"]).exists():
            result.append(p)

    for p in project_paths:
        if Path(p["path"]).exists():
            result.append(p)

    return result
