"""Add patients table, case_reports table, and project fields for doctor portal.

Revision ID: 002
Revises: 5f1fd04def23
Create Date: 2026-02-03
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "5f1fd04def23"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- patients ---
    op.create_table(
        "patients",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("patient_code", sa.String(50), unique=True, nullable=False),
        sa.Column("first_name", sa.String(100), nullable=True),
        sa.Column("last_name", sa.String(100), nullable=True),
        sa.Column("date_of_birth", sa.Date, nullable=True),
        sa.Column("sex", sa.String(20), nullable=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("medical_record_number", sa.String(100), nullable=True),
        sa.Column("insurance_provider", sa.String(255), nullable=True),
        sa.Column("insurance_policy_number", sa.String(100), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_patients_patient_code", "patients", ["patient_code"])

    # --- Add new columns to projects ---
    op.add_column(
        "projects", sa.Column("patient_id", sa.String(36), nullable=True)
    )
    op.add_column(
        "projects", sa.Column("reconstruction_type", sa.String(100), nullable=True)
    )
    op.add_column(
        "projects", sa.Column("implant_material", sa.String(100), nullable=True)
    )
    op.add_column("projects", sa.Column("notes", sa.Text, nullable=True))
    op.add_column(
        "projects", sa.Column("region_code", sa.String(10), nullable=True)
    )
    op.add_column("projects", sa.Column("metadata_json", sa.Text, nullable=True))
    # Note: SQLite doesn't enforce foreign keys by default, so we skip the FK constraint

    # --- case_reports ---
    op.create_table(
        "case_reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "project_id",
            sa.String(36),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("html_content", sa.Text, nullable=False),
        sa.Column("pdf_path", sa.Text, nullable=True),
        sa.Column("template_version", sa.String(50), nullable=False, server_default="v1.0"),
        sa.Column("prompt_version", sa.String(50), nullable=True),
        sa.Column("ai_model", sa.String(100), nullable=True),
        sa.Column("ai_provider", sa.String(50), nullable=True),
        sa.Column("ai_request_id", sa.String(100), nullable=True),
        sa.Column("region_code", sa.String(10), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("metadata_json", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(100), nullable=False, server_default="system"),
    )
    op.create_index("ix_case_reports_project_id", "case_reports", ["project_id"])


def downgrade() -> None:
    op.drop_table("case_reports")
    op.drop_column("projects", "metadata_json")
    op.drop_column("projects", "region_code")
    op.drop_column("projects", "notes")
    op.drop_column("projects", "implant_material")
    op.drop_column("projects", "reconstruction_type")
    op.drop_column("projects", "patient_id")
    op.drop_table("patients")
