/**
 * API service for cran-2 implant generation jobs.
 */
import { apiV1 } from './api-v1';
import type { GenerationJob, GenerationJobCreate } from '../types/generation';

export const generationApi = {
  /** Create a cran-2 generation job and queue it for execution. */
  async createJob(body: GenerationJobCreate): Promise<GenerationJob> {
    const { data } = await apiV1.post<GenerationJob>('/generation-jobs/', body);
    return data;
  },

  async getJob(jobId: string): Promise<GenerationJob> {
    const { data } = await apiV1.get<GenerationJob>(`/generation-jobs/${jobId}`);
    return data;
  },

  async listJobs(params?: {
    project_id?: string;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<GenerationJob[]> {
    const { data } = await apiV1.get<GenerationJob[]>('/generation-jobs/', { params });
    return data;
  },

  async cancelJob(jobId: string): Promise<void> {
    await apiV1.post(`/generation-jobs/${jobId}/cancel`);
  },

  async listArtifacts(jobId: string): Promise<{ job_id: string; artifacts: Array<{ format: string; key: string }> }> {
    const { data } = await apiV1.get(`/generation-jobs/${jobId}/artifacts`);
    return data;
  },

  downloadUrl(jobId: string, format: string): string {
    return `${apiV1.defaults.baseURL}/generation-jobs/${jobId}/download/${format}`;
  },
};
