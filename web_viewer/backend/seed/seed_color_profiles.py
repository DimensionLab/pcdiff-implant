"""Seed default SDF color profiles into the database."""

import json
import logging

from sqlalchemy.orm import Session

from web_viewer.backend.models.color_profile import ColorProfile

logger = logging.getLogger(__name__)

DEFAULT_PROFILES = [
    {
        "name": "Blue-White-Red (Diverging)",
        "description": (
            "Standard diverging colormap. Blue = inside surface (collision), "
            "White = on surface (perfect fit), Red = outside surface (gap)."
        ),
        "color_map_type": "diverging",
        "sdf_range_min": -5.0,
        "sdf_range_max": 5.0,
        "color_stops": json.dumps(
            [
                {"value": 0.0, "color": "#0000ff"},
                {"value": 0.25, "color": "#6666ff"},
                {"value": 0.5, "color": "#ffffff"},
                {"value": 0.75, "color": "#ff6666"},
                {"value": 1.0, "color": "#ff0000"},
            ]
        ),
        "is_default": True,
    },
    {
        "name": "Viridis (Sequential)",
        "description": "Perceptually uniform sequential colormap for distance magnitude.",
        "color_map_type": "sequential",
        "sdf_range_min": 0.0,
        "sdf_range_max": 10.0,
        "color_stops": json.dumps(
            [
                {"value": 0.0, "color": "#440154"},
                {"value": 0.25, "color": "#3b528b"},
                {"value": 0.5, "color": "#21918c"},
                {"value": 0.75, "color": "#5ec962"},
                {"value": 1.0, "color": "#fde725"},
            ]
        ),
        "is_default": False,
    },
    {
        "name": "Clinical Fit Assessment",
        "description": (
            "Green = good fit (near surface), Yellow/Orange = marginal, "
            "Red = poor fit. Designed for implant-to-skull fit assessment."
        ),
        "color_map_type": "diverging",
        "sdf_range_min": -3.0,
        "sdf_range_max": 3.0,
        "color_stops": json.dumps(
            [
                {"value": 0.0, "color": "#ff0000"},
                {"value": 0.35, "color": "#ffaa00"},
                {"value": 0.5, "color": "#00cc00"},
                {"value": 0.65, "color": "#ffaa00"},
                {"value": 1.0, "color": "#ff0000"},
            ]
        ),
        "is_default": False,
    },
]


def seed_color_profiles(db: Session) -> int:
    """Insert default color profiles. Returns number of profiles created."""
    created = 0
    for profile_data in DEFAULT_PROFILES:
        existing = db.query(ColorProfile).filter(ColorProfile.name == profile_data["name"]).first()
        if existing:
            continue

        profile = ColorProfile(**profile_data)
        db.add(profile)
        created += 1

    db.commit()
    logger.info("Seeded %d color profiles", created)
    return created
