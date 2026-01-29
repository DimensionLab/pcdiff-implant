import { apiV1, viewerUrl } from './api-v1';
import type { PointCloud, PointCloudCreate, PointCloudUpdate } from '../types/point-cloud';
import type { STLStatusResponse } from '../types/checker';

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

  // ── STL Mesh Generation ────────────────────────────────

  /** Trigger STL mesh generation from a point cloud */
  async generateSTL(pcId: string, method: string = 'poisson', depth: number = 8): Promise<PointCloud> {
    const { data } = await apiV1.post<PointCloud>(
      `/point-clouds/${pcId}/generate-stl`,
      null,
      { params: { method, depth }, timeout: 180_000 },
    );
    return data;
  },

  /** Check if STL already exists for a point cloud */
  async getSTLStatus(pcId: string): Promise<STLStatusResponse> {
    const { data } = await apiV1.get<STLStatusResponse>(`/point-clouds/${pcId}/stl-status`);
    return data;
  },

  /** Fetch raw STL binary data for Three.js rendering */
  async loadSTL(pcId: string): Promise<ArrayBuffer> {
    const { data } = await apiV1.get(`/point-clouds/${pcId}/stl`, {
      responseType: 'arraybuffer',
    });
    return data;
  },

  /** Get direct download URL for an STL file */
  stlDownloadUrl(pcId: string): string {
    const base = import.meta.env.VITE_API_V1_URL || '/api/v1';
    return `${base}/point-clouds/${pcId}/stl`;
  },
};
