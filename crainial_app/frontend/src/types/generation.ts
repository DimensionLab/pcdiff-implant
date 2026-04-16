/**
 * Types for cran-2 implant generation jobs.
 *
 * A job submits a defective-skull NRRD scan to the cran-2 RunPod endpoint
 * and stores the resulting implant-mask NRRD as a new Scan.
 */

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

export interface GenerationJob {
  id: string;
  name: string;
  description?: string;
  project_id?: string;
  input_scan_id: string;
  output_scan_id?: string;
  status: JobStatus;
  progress_percent: number;
  current_step?: string;
  error_message?: string;
  threshold: number;
  runpod_job_id?: string;
  metrics?: Record<string, unknown> | null;
  queued_at?: string;
  started_at?: string;
  completed_at?: string;
  generation_time_ms?: number;
  inference_time_ms?: number;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface GenerationJobCreate {
  input_scan_id: string;
  threshold?: number;
  project_id?: string;
  name?: string;
  description?: string;
}
