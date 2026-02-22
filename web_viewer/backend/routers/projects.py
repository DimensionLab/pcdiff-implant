"""Project CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.fit_metrics import SkullImplantPair
from web_viewer.backend.schemas.point_cloud import PointCloudRead
from web_viewer.backend.schemas.project import ProjectCreate, ProjectRead, ProjectUpdate
from web_viewer.backend.schemas.scan import ScanRead
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.fit_metrics_service import FitMetricsService
from web_viewer.backend.services.point_cloud_service import PointCloudService
from web_viewer.backend.services.project_service import ProjectService
from web_viewer.backend.services.scan_service import ScanService

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])


def _get_services(db: Session = Depends(get_db)):
    audit = AuditService(db)
    return {
        "project": ProjectService(db, audit),
        "scan": ScanService(db, audit),
        "point_cloud": PointCloudService(db, audit),
        "fit_metrics": FitMetricsService(db, audit),
    }


@router.get("/", response_model=list[ProjectRead])
def list_projects(services=Depends(_get_services)):
    return services["project"].list_projects()


@router.post("/", response_model=ProjectRead, status_code=201)
def create_project(body: ProjectCreate, services=Depends(_get_services)):
    return services["project"].create_project(
        name=body.name,
        description=body.description,
        patient_id=body.patient_id,
        reconstruction_type=body.reconstruction_type,
        implant_material=body.implant_material,
        notes=body.notes,
        region_code=body.region_code,
        metadata_json=body.metadata_json,
    )


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: str, services=Depends(_get_services)):
    project = services["project"].get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: str, body: ProjectUpdate, services=Depends(_get_services)
):
    project = services["project"].update_project(
        project_id, **body.model_dump(exclude_none=True)
    )
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, services=Depends(_get_services)):
    if not services["project"].delete_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")


@router.get("/{project_id}/scans", response_model=list[ScanRead])
def list_project_scans(project_id: str, services=Depends(_get_services)):
    return services["scan"].list_scans(project_id=project_id)


@router.get("/{project_id}/point-clouds", response_model=list[PointCloudRead])
def list_project_point_clouds(project_id: str, services=Depends(_get_services)):
    return services["point_cloud"].list_point_clouds(project_id=project_id)


@router.get("/{project_id}/auto-match", response_model=list[SkullImplantPair])
def auto_match_point_clouds(project_id: str, services=Depends(_get_services)):
    """Find skull/implant pairs within a project, grouped by skull_id.

    Returns pairs where at least one defective_skull and one
    implant/generated_implant share the same skull_id.
    """
    project = services["project"].get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return services["fit_metrics"].auto_match_by_skull_id(project_id)
