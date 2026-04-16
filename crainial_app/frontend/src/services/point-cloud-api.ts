import { apiV1, viewerUrl } from './api-v1';
import type { PointCloud, PointCloudCreate, PointCloudUpdate } from '../types/point-cloud';

export const pointCloudApi = {
  async list(params?: {
    project_id?: string;
    scan_id?: string;
    scan_category?: string;
    skull_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<PointCloud[]> {
    const { data } = await apiV1.get<PointCloud[]>('/point-clouds/', { params });
    return data;
  },

  async get(pcId: string): Promise<PointCloud> {
    const { data } = await apiV1.get<PointCloud>(`/point-clouds/${pcId}`);
    return data;
  },

  async create(body: PointCloudCreate): Promise<PointCloud> {
    const { data } = await apiV1.post<PointCloud>('/point-clouds/', body);
    return data;
  },

  async update(pcId: string, body: PointCloudUpdate): Promise<PointCloud> {
    const { data } = await apiV1.put<PointCloud>(`/point-clouds/${pcId}`, body);
    return data;
  },

  async delete(pcId: string): Promise<void> {
    await apiV1.delete(`/point-clouds/${pcId}`);
  },

  /** Fetch raw NPY data as ArrayBuffer for Three.js rendering */
  async loadData(pcId: string): Promise<ArrayBuffer> {
    const { data } = await apiV1.get(`/point-clouds/${pcId}/data`, {
      responseType: 'arraybuffer',
    });
    return data;
  },

  /** Fetch SDF values as Float32Array */
  async loadSDF(pcId: string): Promise<Float32Array> {
    const { data } = await apiV1.get(`/point-clouds/${pcId}/sdf`, {
      responseType: 'arraybuffer',
    });
    return new Float32Array(data);
  },

  /** URL for raw NPY file serving */
  npyUrl(pcId: string): string {
    return viewerUrl(`/point-clouds/${pcId}/npy`);
  },

};
