"""cran-2 refactor: replace generation_jobs schema for cran-2 NRRD pipeline.

Drops the old PCDiff/voxelization columns and replaces them with the cran-2
shape: input_scan_id + output_scan_id + threshold + runpod_job_id.

Revision ID: 006
Revises: 004
Create Date: 2026-04-16
"""

import sqlalchemy as sa
from alembic import op

revision = "006"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The cran-2 schema is a clean break from the PCDiff/voxelization schema:
    # different inputs (Scan NRRD vs PointCloud npy), different outputs (single
    # Scan NRRD vs ensemble PointClouds + STL meshes), and no parent-child jobs.
    # We drop and recreate rather than chain a dozen alters.
    op.drop_table("generation_jobs")

    op.create_table(
        "generation_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "input_scan_id",
            sa.String(36),
            sa.ForeignKey("scans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "output_scan_id",
            sa.String(36),
            sa.ForeignKey("scans.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("progress_percent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_step", sa.String(255), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("threshold", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("runpod_job_id", sa.String(100), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("generation_time_ms", sa.Integer(), nullable=True),
        sa.Column("inference_time_ms", sa.Integer(), nullable=True),
        sa.Column("metrics_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_generation_jobs_status", "generation_jobs", ["status"])
    op.create_index("ix_generation_jobs_project_id", "generation_jobs", ["project_id"])
    op.create_index("ix_generation_jobs_input_scan_id", "generation_jobs", ["input_scan_id"])

    # Replace the legacy PCDiff/voxelization settings with cran-2 settings.
    settings = sa.table(
        "settings",
        sa.column("key", sa.String(100)),
    )
    op.execute(
        settings.delete().where(
            settings.c.key.in_(
                [
                    "default_sampling_method",
                    "default_sampling_steps",
                    "default_ensemble_count",
                    "pcdiff_model_path",
                    "voxelization_model_path",
                    "default_voxelization_resolution",
                    "default_smoothing_iterations",
                    "default_close_holes",
                ]
            )
        )
    )


def downgrade() -> None:
    op.drop_index("ix_generation_jobs_input_scan_id", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_project_id", table_name="generation_jobs")
    op.drop_index("ix_generation_jobs_status", table_name="generation_jobs")
    op.drop_table("generation_jobs")
