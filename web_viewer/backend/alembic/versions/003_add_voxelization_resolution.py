"""Add voxelization_resolution and source_implant_pc_id to generation_jobs.

Revision ID: 003_add_voxelization_resolution
Revises: 002_add_patients_case_reports_project_fields
Create Date: 2026-02-03

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add voxelization_resolution column with default value of 512
    op.add_column(
        "generation_jobs",
        sa.Column("voxelization_resolution", sa.Integer(), nullable=False, server_default="512"),
    )
    
    # Add source_implant_pc_id for re-voxelization jobs
    # Note: SQLite doesn't enforce foreign keys by default, so we skip the FK constraint
    op.add_column(
        "generation_jobs",
        sa.Column("source_implant_pc_id", sa.String(36), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("generation_jobs", "source_implant_pc_id")
    op.drop_column("generation_jobs", "voxelization_resolution")
