"""ColorProfile CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from crainial_app.backend.database import get_db
from crainial_app.backend.schemas.color_profile import (
    ColorProfileCreate,
    ColorProfileRead,
    ColorProfileUpdate,
)
from crainial_app.backend.services.audit_service import AuditService
from crainial_app.backend.services.color_profile_service import ColorProfileService

router = APIRouter(prefix="/api/v1/color-profiles", tags=["color-profiles"])


def _get_service(db: Session = Depends(get_db)):
    return ColorProfileService(db, AuditService(db))


@router.get("/", response_model=list[ColorProfileRead])
def list_profiles(service: ColorProfileService = Depends(_get_service)):
    return service.list_profiles()


@router.post("/", response_model=ColorProfileRead, status_code=201)
def create_profile(body: ColorProfileCreate, service: ColorProfileService = Depends(_get_service)):
    return service.create_profile(
        name=body.name,
        description=body.description,
        color_map_type=body.color_map_type,
        color_stops=body.color_stops,
        sdf_range_min=body.sdf_range_min,
        sdf_range_max=body.sdf_range_max,
        is_default=body.is_default,
    )


@router.get("/default", response_model=ColorProfileRead)
def get_default_profile(service: ColorProfileService = Depends(_get_service)):
    profile = service.get_default_profile()
    if not profile:
        raise HTTPException(status_code=404, detail="No default profile set")
    return profile


@router.get("/{profile_id}", response_model=ColorProfileRead)
def get_profile(profile_id: str, service: ColorProfileService = Depends(_get_service)):
    profile = service.get_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="ColorProfile not found")
    return profile


@router.put("/{profile_id}", response_model=ColorProfileRead)
def update_profile(
    profile_id: str,
    body: ColorProfileUpdate,
    service: ColorProfileService = Depends(_get_service),
):
    profile = service.update_profile(profile_id, **body.model_dump(exclude_none=True))
    if not profile:
        raise HTTPException(status_code=404, detail="ColorProfile not found")
    return profile


@router.delete("/{profile_id}", status_code=204)
def delete_profile(profile_id: str, service: ColorProfileService = Depends(_get_service)):
    if not service.delete_profile(profile_id):
        raise HTTPException(status_code=404, detail="ColorProfile not found")
