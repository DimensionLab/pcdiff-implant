"""Add smoothing_iterations and close_holes to generation_jobs.

Revision ID: 004_add_smoothing_and_close_holes
Revises: 003
Create Date: 2026-02-17

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generation_jobs",
        sa.Column("smoothing_iterations", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("close_holes", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade() -> None:
    op.drop_column("generation_jobs", "close_holes")
    op.drop_column("generation_jobs", "smoothing_iterations")
