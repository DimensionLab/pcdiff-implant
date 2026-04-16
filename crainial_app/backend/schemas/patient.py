"""Pydantic schemas for Patient endpoints."""

from datetime import date, datetime

from pydantic import BaseModel


class PatientCreate(BaseModel):
    """Request to create a new patient."""

    patient_code: str
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    sex: str | None = None
    email: str | None = None
    phone: str | None = None
    medical_record_number: str | None = None
    insurance_provider: str | None = None
    insurance_policy_number: str | None = None
    notes: str | None = None
    metadata_json: str | None = None


class PatientUpdate(BaseModel):
    """Request to update a patient."""

    patient_code: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    sex: str | None = None
    email: str | None = None
    phone: str | None = None
    medical_record_number: str | None = None
    insurance_provider: str | None = None
    insurance_policy_number: str | None = None
    notes: str | None = None
    metadata_json: str | None = None


class PatientRead(BaseModel):
    """Response schema for patient."""

    id: str
    patient_code: str
    first_name: str | None = None
    last_name: str | None = None
    date_of_birth: date | None = None
    sex: str | None = None
    email: str | None = None
    phone: str | None = None
    medical_record_number: str | None = None
    insurance_provider: str | None = None
    insurance_policy_number: str | None = None
    notes: str | None = None
    metadata_json: str | None = None
    created_at: datetime
    updated_at: datetime
    created_by: str

    model_config = {"from_attributes": True}
