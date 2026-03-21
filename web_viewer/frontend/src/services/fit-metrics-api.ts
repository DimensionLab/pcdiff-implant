import { apiV1 } from './api-v1';
import type { FitMetricsResult, FitMetricsRequest, SkullImplantPair } from '../types/checker';
import type { Project, ProjectCreate, ProjectUpdate } from '../types/project';
import type { PointCloud } from '../types/point-cloud';

export const fitMetricsApi = {
  async compute(body: FitMetricsRequest): Promise<FitMetricsResult> {
    const { data } = await apiV1.post<FitMetricsResult>('/fit-metrics/compute', body);
    return data;
  },

  async get(id: string): Promise<FitMetricsResult> {
    const { data } = await apiV1.get<FitMetricsResult>(`/fit-metrics/${id}`);
    return data;
  },

  async list(params?: {
    implant_pc_id?: string;
    reference_pc_id?: string;
  }): Promise<FitMetricsResult[]> {
    const { data } = await apiV1.get<FitMetricsResult[]>('/fit-metrics/', { params });
    return data;
  },

  async computeSDFHeatmap(queryPcId: string, referencePcId: string): Promise<Float32Array> {
    const { data } = await apiV1.post(
      '/fit-metrics/sdf-heatmap',
      { query_pc_id: queryPcId, reference_pc_id: referencePcId },
      { responseType: 'arraybuffer' },
    );
    return new Float32Array(data);
  },
};

export const projectApi = {
  async list(): Promise<Project[]> {
    const { data } = await apiV1.get<Project[]>('/projects/');
    return data;
  },

  async get(id: string): Promise<Project> {
    const { data } = await apiV1.get<Project>(`/projects/${id}`);
    return data;
  },

  async create(body: ProjectCreate): Promise<Project> {
    const { data } = await apiV1.post<Project>('/projects/', body);
    return data;
  },

  async update(id: string, body: ProjectUpdate): Promise<Project> {
    const { data } = await apiV1.put<Project>(`/projects/${id}`, body);
    return data;
  },

  async delete(id: string): Promise<void> {
    await apiV1.delete(`/projects/${id}`);
  },

  async getPointClouds(projectId: string): Promise<PointCloud[]> {
    const { data } = await apiV1.get<PointCloud[]>(`/projects/${projectId}/point-clouds`);
    return data;
  },

  async getAutoMatch(projectId: string): Promise<SkullImplantPair[]> {
    const { data } = await apiV1.get<SkullImplantPair[]>(`/projects/${projectId}/auto-match`);
    return data;
  },
};
