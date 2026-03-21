/**
 * API service for implant generation jobs.
 * 
 * Supports parallel ensemble generation:
 * - When num_ensemble > 1 with cloud, creates parent job with N child jobs
 * - getJob() returns GenerationJobWithChildren including child_jobs array
 */
import { apiV1 } from './api-v1';
import type { GenerationJob, GenerationJobCreate, GenerationJobWithChildren, RevoxelizeJobCreate } from '../types/generation';

export const generationApi = {
  /** Create a new generation job and queue it for execution.
   * 
   * For cloud generation with num_ensemble > 1:
   * - Creates a parent job with N child jobs running in parallel
   * - Returns the parent job immediately
   */
  async createJob(body: GenerationJobCreate): Promise<GenerationJob> {
    const { data } = await apiV1.post<GenerationJob>('/generation-jobs/', body);
    return data;
  },

  /** Get a specific generation job with its child jobs (if any).
   * 
   * For parent jobs, includes child_jobs array with status of each parallel ensemble.
   */
  async getJob(jobId: string): Promise<GenerationJobWithChildren> {
    const { data } = await apiV1.get<GenerationJobWithChildren>(`/generation-jobs/${jobId}`);
    return data;
  },

  /** Get child jobs for a parent job */
  async getChildJobs(jobId: string): Promise<GenerationJob[]> {
    const { data } = await apiV1.get<GenerationJob[]>(`/generation-jobs/${jobId}/children`);
    return data;
  },

  /** List generation jobs with optional filtering.
   * 
   * By default only returns parent jobs (not child jobs).
   */
  async listJobs(params?: {
    project_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<GenerationJob[]> {
    const { data } = await apiV1.get<GenerationJob[]>('/generation-jobs/', { params });
    return data;
  },

  /** Cancel a pending or running job */
  async cancelJob(jobId: string): Promise<void> {
    await apiV1.post(`/generation-jobs/${jobId}/cancel`);
  },

  /** Select a specific output from ensemble results */
  async selectOutput(jobId: string, outputId: string): Promise<GenerationJob> {
    const { data } = await apiV1.post<GenerationJob>(
      `/generation-jobs/${jobId}/select-output`,
      { output_id: outputId },
    );
    return data;
  },

  /** Delete all unselected outputs from a completed job */
  async deleteUnselectedOutputs(jobId: string): Promise<void> {
    await apiV1.delete(`/generation-jobs/${jobId}/unselected-outputs`);
  },

  /** Create a re-voxelization job to regenerate mesh with different resolution.
   * 
   * Use this to generate a new STL mesh with a different level of detail
   * from an already-generated implant point cloud.
   * 
   * Resolution options:
   * - 128: Fast, low detail (for previews)
   * - 256: Medium detail
   * - 512: High detail (default, balanced)
   * - 1024: Ultra detail (slower, for final production)
   */
  async createRevoxelizationJob(body: RevoxelizeJobCreate): Promise<GenerationJob> {
    const { data } = await apiV1.post<GenerationJob>('/generation-jobs/revoxelize', body);
    return data;
  },
};
