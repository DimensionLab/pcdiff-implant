export interface Project {
  id: string;
  name: string;
  description: string | null;
  patient_id: string | null;
  reconstruction_type: string | null;
  implant_material: string | null;
  notes: string | null;
  region_code: string | null;
  metadata_json: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface ProjectCreate {
  name: string;
  description?: string;
  patient_id?: string;
  reconstruction_type?: string;
  implant_material?: string;
  notes?: string;
  region_code?: string;
  metadata_json?: string;
}

export interface ProjectUpdate {
  name?: string;
  description?: string;
  patient_id?: string;
  reconstruction_type?: string;
  implant_material?: string;
  notes?: string;
  region_code?: string;
  metadata_json?: string;
}
