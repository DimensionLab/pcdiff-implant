"""Service for managing ColorProfile records."""

import logging

from sqlalchemy.orm import Session

from web_viewer.backend.models.color_profile import ColorProfile
from web_viewer.backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class ColorProfileService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def list_profiles(self) -> list[ColorProfile]:
        return self.db.query(ColorProfile).order_by(ColorProfile.name).all()

    def get_profile(self, profile_id: str) -> ColorProfile | None:
        return self.db.query(ColorProfile).filter(ColorProfile.id == profile_id).first()

    def get_default_profile(self) -> ColorProfile | None:
        return (
            self.db.query(ColorProfile)
            .filter(ColorProfile.is_default.is_(True))
            .first()
        )

    def create_profile(
        self,
        name: str,
        color_map_type: str,
        color_stops: str,
        description: str | None = None,
        sdf_range_min: float = -1.0,
        sdf_range_max: float = 1.0,
        is_default: bool = False,
    ) -> ColorProfile:
        # If setting as default, unset existing default
        if is_default:
            self._clear_defaults()

        profile = ColorProfile(
            name=name,
            description=description,
            color_map_type=color_map_type,
            color_stops=color_stops,
            sdf_range_min=sdf_range_min,
            sdf_range_max=sdf_range_max,
            is_default=is_default,
        )
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)

        self.audit.log(
            action="color_profile.create",
            entity_type="color_profile",
            entity_id=profile.id,
            details={"name": name},
        )
        return profile

    def update_profile(self, profile_id: str, **kwargs) -> ColorProfile | None:
        profile = self.get_profile(profile_id)
        if not profile:
            return None

        if kwargs.get("is_default"):
            self._clear_defaults()

        for key, value in kwargs.items():
            if hasattr(profile, key) and value is not None:
                setattr(profile, key, value)

        self.db.commit()
        self.db.refresh(profile)

        self.audit.log(
            action="color_profile.update",
            entity_type="color_profile",
            entity_id=profile_id,
            details={"updated_fields": list(kwargs.keys())},
        )
        return profile

    def delete_profile(self, profile_id: str) -> bool:
        profile = self.get_profile(profile_id)
        if not profile:
            return False
        self.db.delete(profile)
        self.db.commit()
        self.audit.log(
            action="color_profile.delete",
            entity_type="color_profile",
            entity_id=profile_id,
        )
        return True

    def _clear_defaults(self) -> None:
        self.db.query(ColorProfile).filter(ColorProfile.is_default.is_(True)).update(
            {"is_default": False}
        )
        self.db.flush()
