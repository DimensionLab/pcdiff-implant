/**
 * Types for implant generation jobs.
 */

export interface GenerationJob {
  id: string;
  name: string;
  description?: string;
  project_id?: string;
  input_pc_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percent: number;
  current_step?: string;
  error_message?: string;
  sampling_method: 'ddim' | 'ddpm';
  sampling_steps: number;
  num_ensemble: number;
  output_pc_ids: string[];
  output_stl_ids: string[];
  selected_output_id?: string;
  metrics?: Record<string, { dsc?: number; bdsc?: number; hd95?: number }>;
  queued_at?: string;
  started_at?: string;
  completed_at?: string;
  generation_time_ms?: number;
  created_at: string;
  updated_at: string;
  created_by: string;
}

export interface GenerationJobCreate {
  project_id: string;
  input_pc_id: string;
  sampling_method?: 'ddim' | 'ddpm';
  sampling_steps?: number;
  num_ensemble?: number;
  name?: string;
  description?: string;
}
