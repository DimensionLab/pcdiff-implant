/**
 * Types for case reports in the doctor portal.
 */

export interface CaseReport {
  id: string;
  project_id: string;
  title: string;
  html_content: string;
  pdf_path: string | null;
  template_version: string;
  prompt_version: string | null;
  ai_model: string | null;
  ai_provider: string | null;
  ai_request_id: string | null;
  region_code: string | null;
  generated_at: string;
  metadata_json: string | null;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface CaseReportSummary {
  id: string;
  project_id: string;
  title: string;
  template_version: string;
  ai_model: string | null;
  region_code: string | null;
  generated_at: string;
  created_at: string;
}

export interface CaseReportCreate {
  project_id: string;
  title?: string;
  region_code?: string;
}
