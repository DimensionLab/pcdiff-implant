"""Service for managing Patient records."""

import logging
from datetime import date

from sqlalchemy.orm import Session

from crainial_app.backend.models.patient import Patient
from crainial_app.backend.services.audit_service import AuditService

logger = logging.getLogger(__name__)


class PatientService:
    def __init__(self, db: Session, audit: AuditService):
        self.db = db
        self.audit = audit

    def list_patients(
        self,
        limit: int = 100,
        offset: int = 0,
        search: str | None = None,
    ) -> list[Patient]:
        """List patients with optional search filtering."""
        query = self.db.query(Patient)

        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                Patient.patient_code.ilike(search_pattern)
                | Patient.first_name.ilike(search_pattern)
                | Patient.last_name.ilike(search_pattern)
                | Patient.medical_record_number.ilike(search_pattern)
            )

        return query.order_by(Patient.created_at.desc()).offset(offset).limit(limit).all()

    def get_patient(self, patient_id: str) -> Patient | None:
        """Get a patient by ID."""
        return self.db.query(Patient).filter(Patient.id == patient_id).first()

    def get_patient_by_code(self, patient_code: str) -> Patient | None:
        """Get a patient by their unique code."""
        return self.db.query(Patient).filter(Patient.patient_code == patient_code).first()

    def create_patient(
        self,
        patient_code: str,
        first_name: str | None = None,
        last_name: str | None = None,
        date_of_birth: date | None = None,
        sex: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        medical_record_number: str | None = None,
        insurance_provider: str | None = None,
        insurance_policy_number: str | None = None,
        notes: str | None = None,
        metadata_json: str | None = None,
    ) -> Patient:
        """Create a new patient record."""
        # Check for duplicate patient_code
        existing = self.get_patient_by_code(patient_code)
        if existing:
            raise ValueError(f"Patient with code '{patient_code}' already exists")

        patient = Patient(
            patient_code=patient_code,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=date_of_birth,
            sex=sex,
            email=email,
            phone=phone,
            medical_record_number=medical_record_number,
            insurance_provider=insurance_provider,
            insurance_policy_number=insurance_policy_number,
            notes=notes,
            metadata_json=metadata_json,
        )
        self.db.add(patient)
        self.db.commit()
        self.db.refresh(patient)

        self.audit.log(
            action="patient.create",
            entity_type="patient",
            entity_id=patient.id,
            details={"patient_code": patient_code},
        )

        logger.info(f"Created patient: {patient.id} ({patient_code})")
        return patient

    def update_patient(self, patient_id: str, **kwargs) -> Patient | None:
        """Update an existing patient record."""
        patient = self.get_patient(patient_id)
        if not patient:
            return None

        # Check for duplicate patient_code if being changed
        new_code = kwargs.get("patient_code")
        if new_code and new_code != patient.patient_code:
            existing = self.get_patient_by_code(new_code)
            if existing:
                raise ValueError(f"Patient with code '{new_code}' already exists")

        updated_fields = []
        for key, value in kwargs.items():
            if hasattr(patient, key) and value is not None:
                setattr(patient, key, value)
                updated_fields.append(key)

        self.db.commit()
        self.db.refresh(patient)

        self.audit.log(
            action="patient.update",
            entity_type="patient",
            entity_id=patient_id,
            details={"updated_fields": updated_fields},
        )

        logger.info(f"Updated patient: {patient_id}")
        return patient

    def delete_patient(self, patient_id: str) -> bool:
        """Delete a patient record."""
        patient = self.get_patient(patient_id)
        if not patient:
            return False

        patient_code = patient.patient_code
        self.db.delete(patient)
        self.db.commit()

        self.audit.log(
            action="patient.delete",
            entity_type="patient",
            entity_id=patient_id,
            details={"patient_code": patient_code},
        )

        logger.info(f"Deleted patient: {patient_id} ({patient_code})")
        return True
