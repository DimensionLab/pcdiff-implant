"""Service for managing Project records."""

import logging

from sqlalchemy.orm import Session

from web_viewer.backend.models.project import Project
from web_viewer.backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class ProjectService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def list_projects(self, limit: int = 100, offset: int = 0) -> list[Project]:
        return self.db.query(Project).order_by(Project.created_at.desc()).offset(offset).limit(limit).all()

    def get_project(self, project_id: str) -> Project | None:
        return self.db.query(Project).filter(Project.id == project_id).first()

    def create_project(
        self,
        name: str,
        description: str | None = None,
        patient_id: str | None = None,
        reconstruction_type: str | None = None,
        implant_material: str | None = None,
        notes: str | None = None,
        region_code: str | None = None,
        metadata_json: str | None = None,
    ) -> Project:
        project = Project(
            name=name,
            description=description,
            patient_id=patient_id,
            reconstruction_type=reconstruction_type,
            implant_material=implant_material,
            notes=notes,
            region_code=region_code,
            metadata_json=metadata_json,
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)

        self.audit.log(
            action="project.create",
            entity_type="project",
            entity_id=project.id,
            details={"name": name, "patient_id": patient_id},
        )
        return project

    def update_project(self, project_id: str, **kwargs) -> Project | None:
        project = self.get_project(project_id)
        if not project:
            return None

        for key, value in kwargs.items():
            if hasattr(project, key) and value is not None:
                setattr(project, key, value)

        self.db.commit()
        self.db.refresh(project)

        self.audit.log(
            action="project.update",
            entity_type="project",
            entity_id=project_id,
            details={"updated_fields": list(kwargs.keys())},
        )
        return project

    def delete_project(self, project_id: str) -> bool:
        project = self.get_project(project_id)
        if not project:
            return False
        self.db.delete(project)
        self.db.commit()
        self.audit.log(
            action="project.delete",
            entity_type="project",
            entity_id=project_id,
        )
        return True
