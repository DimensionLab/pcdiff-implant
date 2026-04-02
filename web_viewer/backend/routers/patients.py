"""Patient CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from web_viewer.backend.database import get_db
from web_viewer.backend.schemas.patient import PatientCreate, PatientRead, PatientUpdate
from web_viewer.backend.schemas.project import ProjectRead
from web_viewer.backend.services.audit_service import AuditService
from web_viewer.backend.services.patient_service import PatientService
from web_viewer.backend.services.project_service import ProjectService

router = APIRouter(prefix="/api/v1/patients", tags=["patients"])


def _get_services(db: Session = Depends(get_db)):
    audit = AuditService(db)
    return {
        "patient": PatientService(db, audit),
        "project": ProjectService(db, audit),
    }


@router.get("/", response_model=list[PatientRead])
def list_patients(
    search: str | None = Query(None, description="Search by code, name, or MRN"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    services=Depends(_get_services),
):
    """List all patients with optional search filtering."""
    return services["patient"].list_patients(
        search=search,
        limit=limit,
        offset=offset,
    )


@router.post("/", response_model=PatientRead, status_code=201)
def create_patient(body: PatientCreate, services=Depends(_get_services)):
    """Create a new patient record."""
    try:
        return services["patient"].create_patient(
            patient_code=body.patient_code,
            first_name=body.first_name,
            last_name=body.last_name,
            date_of_birth=body.date_of_birth,
            sex=body.sex,
            email=body.email,
            phone=body.phone,
            medical_record_number=body.medical_record_number,
            insurance_provider=body.insurance_provider,
            insurance_policy_number=body.insurance_policy_number,
            notes=body.notes,
            metadata_json=body.metadata_json,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{patient_id}", response_model=PatientRead)
def get_patient(patient_id: str, services=Depends(_get_services)):
    """Get a specific patient by ID."""
    patient = services["patient"].get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientRead)
def update_patient(patient_id: str, body: PatientUpdate, services=Depends(_get_services)):
    """Update a patient record."""
    try:
        patient = services["patient"].update_patient(patient_id, **body.model_dump(exclude_none=True))
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        return patient
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{patient_id}", status_code=204)
def delete_patient(patient_id: str, services=Depends(_get_services)):
    """Delete a patient record."""
    if not services["patient"].delete_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")


@router.get("/{patient_id}/projects", response_model=list[ProjectRead])
def list_patient_projects(patient_id: str, services=Depends(_get_services)):
    """List all projects/cases for a patient."""
    patient = services["patient"].get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Query projects linked to this patient
    from web_viewer.backend.models.project import Project

    projects = (
        services["project"]
        .db.query(Project)
        .filter(Project.patient_id == patient_id)
        .order_by(Project.created_at.desc())
        .all()
    )
    return projects
