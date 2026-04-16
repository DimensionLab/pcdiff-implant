"""Initial schema: projects, scans, point_clouds, audit_log, color_profiles.

Revision ID: 001
Create Date: 2025-01-28
"""

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- projects ---
    op.create_table(
        "projects",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )

    # --- scans ---
    op.create_table(
        "scans",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("file_format", sa.String(20), nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("volume_dims_x", sa.Integer, nullable=True),
        sa.Column("volume_dims_y", sa.Integer, nullable=True),
        sa.Column("volume_dims_z", sa.Integer, nullable=True),
        sa.Column("voxel_spacing_x", sa.Float, nullable=True),
        sa.Column("voxel_spacing_y", sa.Float, nullable=True),
        sa.Column("voxel_spacing_z", sa.Float, nullable=True),
        sa.Column("scan_category", sa.String(50), nullable=True),
        sa.Column("defect_type", sa.String(50), nullable=True),
        sa.Column("skull_id", sa.String(20), nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("checksum_sha256", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_scans_skull_id", "scans", ["skull_id"])
    op.create_index("ix_scans_scan_category", "scans", ["scan_category"])

    # --- point_clouds ---
    op.create_table(
        "point_clouds",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "scan_id",
            sa.String(36),
            sa.ForeignKey("scans.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("file_format", sa.String(10), nullable=False),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("num_points", sa.Integer, nullable=True),
        sa.Column("point_dims", sa.Integer, nullable=False, server_default="3"),
        sa.Column("scan_category", sa.String(50), nullable=True),
        sa.Column("defect_type", sa.String(50), nullable=True),
        sa.Column("skull_id", sa.String(20), nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("checksum_sha256", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_point_clouds_skull_id", "point_clouds", ["skull_id"])
    op.create_index("ix_point_clouds_scan_category", "point_clouds", ["scan_category"])

    # --- audit_log (append-only) ---
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", sa.String(36), nullable=True),
        sa.Column("user_id", sa.String(100), nullable=False, server_default="system"),
        sa.Column("details_json", sa.Text, nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("software_version", sa.String(20), nullable=False),
    )
    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_entity", "audit_log", ["entity_type", "entity_id"])
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])

    # --- color_profiles ---
    op.create_table(
        "color_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(100), unique=True, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("color_map_type", sa.String(20), nullable=False),
        sa.Column("color_stops", sa.Text, nullable=False),
        sa.Column("sdf_range_min", sa.Float, nullable=False, server_default="-1.0"),
        sa.Column("sdf_range_max", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )


def downgrade() -> None:
    op.drop_table("color_profiles")
    op.drop_table("audit_log")
    op.drop_table("point_clouds")
    op.drop_table("scans")
    op.drop_table("projects")
