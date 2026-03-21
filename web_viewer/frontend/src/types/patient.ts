/**
 * Types for patient records in the doctor portal.
 */

export interface Patient {
  id: string;
  patient_code: string;
  first_name: string | null;
  last_name: string | null;
  date_of_birth: string | null;
  sex: string | null;
  email: string | null;
  phone: string | null;
  medical_record_number: string | null;
  insurance_provider: string | null;
  insurance_policy_number: string | null;
  notes: string | null;
  metadata_json: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface PatientCreate {
  patient_code: string;
  first_name?: string;
  last_name?: string;
  date_of_birth?: string;
  sex?: string;
  email?: string;
  phone?: string;
  medical_record_number?: string;
  insurance_provider?: string;
  insurance_policy_number?: string;
  notes?: string;
  metadata_json?: string;
}

export interface PatientUpdate {
  patient_code?: string;
  first_name?: string;
  last_name?: string;
  date_of_birth?: string;
  sex?: string;
  email?: string;
  phone?: string;
  medical_record_number?: string;
  insurance_provider?: string;
  insurance_policy_number?: string;
  notes?: string;
  metadata_json?: string;
}
