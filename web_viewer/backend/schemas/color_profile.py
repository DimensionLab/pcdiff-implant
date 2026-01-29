"""Pydantic schemas for ColorProfile endpoints."""

from datetime import datetime

from pydantic import BaseModel


class ColorProfileCreate(BaseModel):
    name: str
    description: str | None = None
    color_map_type: str  # 'diverging' | 'sequential' | 'categorical'
    color_stops: str  # JSON array of {"value": float, "color": "#rrggbb"}
    sdf_range_min: float = -1.0
    sdf_range_max: float = 1.0
    is_default: bool = False


class ColorProfileUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color_map_type: str | None = None
    color_stops: str | None = None
    sdf_range_min: float | None = None
    sdf_range_max: float | None = None
    is_default: bool | None = None


class ColorProfileRead(BaseModel):
    id: str
    name: str
    description: str | None = None
    color_map_type: str
    color_stops: str
    sdf_range_min: float
    sdf_range_max: float
    is_default: bool
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}
