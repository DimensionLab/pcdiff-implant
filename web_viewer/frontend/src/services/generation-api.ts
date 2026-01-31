/**
 * API service for implant generation jobs.
 */
import { apiV1 } from './api-v1';
import type { GenerationJob, GenerationJobCreate } from '../types/generation';

export const generationApi = {
  /** Create a new generation job and queue it for execution */
  async createJob(body: GenerationJobCreate): Promise<GenerationJob> {
    const { data } = await apiV1.post<GenerationJob>('/generation-jobs/', body);
    return data;
  },

  /** Get a specific generation job */
  async getJob(jobId: string): Promise<GenerationJob> {
    const { data } = await apiV1.get<GenerationJob>(`/generation-jobs/${jobId}`);
    return data;
  },

  /** List generation jobs with optional filtering */
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
};
