/**
 * Types for implant generation jobs.
 * 
 * Supports parent-child job hierarchy for parallel ensemble generation:
 * - Parent job: Created when user requests N ensembles with cloud, tracks overall progress
 * - Child jobs: Individual jobs (one per ensemble) that run in parallel on separate workers
 */

export type PcdiffModel = 'best' | 'latest';
export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';

/**
 * Voxelization resolution options.
 * Higher = more detail, slower processing, more memory usage.
 */
export type VoxelizationResolution = 128 | 256 | 512 | 1024;

export interface GenerationJob {
  id: string;
  name: string;
  description?: string;
  project_id?: string;
  input_pc_id: string;
  status: JobStatus;
  progress_percent: number;
  current_step?: string;
  error_message?: string;
  sampling_method: 'ddim' | 'ddpm';
  sampling_steps: number;
  num_ensemble: number;
  pcdiff_model?: PcdiffModel;
  voxelization_resolution: VoxelizationResolution;
  smoothing_iterations: number;  // Laplacian smoothing iterations (0 = disabled)
  close_holes: boolean;  // Whether to fill holes in mesh
  source_implant_pc_id?: string;  // Set for re-voxelization jobs
  
  // Parent-child hierarchy fields
  parent_job_id?: string;  // Set for child jobs
  ensemble_index?: number;  // 0-based index for child jobs
  
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

/**
 * Extended job type that includes child jobs for parent jobs.
 * Returned by the GET /generation-jobs/{id} endpoint.
 */
export interface GenerationJobWithChildren extends GenerationJob {
  child_jobs: GenerationJob[];
  
  // Computed fields from backend
  is_parent_job?: boolean;
  overall_progress?: number;
  completed_children?: number;
  failed_children?: number;
}

export interface GenerationJobCreate {
  project_id: string;
  input_pc_id: string;
  sampling_method?: 'ddim' | 'ddpm';
  sampling_steps?: number;
  num_ensemble?: number;
  name?: string;
  description?: string;
  use_cloud?: boolean;  // null = use default from settings, true = force cloud, false = force local
  pcdiff_model?: PcdiffModel;  // "best" or "latest" - which PCDiff model checkpoint to use
  voxelization_resolution?: VoxelizationResolution;  // PSR grid resolution (default: 512)
  smoothing_iterations?: number;  // Laplacian smoothing: 0 = disabled, 1-100
  close_holes?: boolean;  // Fill holes in generated mesh
}

/**
 * Request to create a re-voxelization job.
 * Re-voxelization generates a new mesh from an existing implant point cloud
 * with a different level of detail (resolution).
 */
export interface RevoxelizeJobCreate {
  project_id: string;
  source_implant_pc_id: string;  // Existing implant point cloud to re-voxelize
  input_pc_id: string;  // Defective skull point cloud (needed for combined voxelization)
  voxelization_resolution?: VoxelizationResolution;  // PSR grid resolution (default: 512)
  name?: string;
  description?: string;
  use_cloud?: boolean;
  smoothing_iterations?: number;  // Laplacian smoothing: 0 = disabled, 1-100
  close_holes?: boolean;  // Fill holes in generated mesh
}
