"""Add parent_job_id and ensemble_index to generation_jobs

Revision ID: 5f1fd04def23
Revises: 001
Create Date: 2026-02-03 11:30:38.024825

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5f1fd04def23'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent-child hierarchy fields for parallel ensemble generation
    # Note: SQLite doesn't enforce foreign keys by default, so we skip the FK constraint
    op.add_column('generation_jobs', sa.Column('parent_job_id', sa.String(length=36), nullable=True))
    op.add_column('generation_jobs', sa.Column('ensemble_index', sa.Integer(), nullable=True))
    # Add pcdiff_model field for model selection
    op.add_column('generation_jobs', sa.Column('pcdiff_model', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('generation_jobs', 'pcdiff_model')
    op.drop_column('generation_jobs', 'ensemble_index')
    op.drop_column('generation_jobs', 'parent_job_id')
