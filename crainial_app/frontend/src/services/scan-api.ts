import { apiV1, viewerUrl } from './api-v1';
import type { Scan, ScanCreate, ScanUpdate, SkullBreakImportRequest, ImportResult, VolumeMetadata } from '../types/scan';
import type { PointCloud } from '../types/point-cloud';

export const scanApi = {
  async list(params?: {
    project_id?: string;
    scan_category?: string;
    skull_id?: string;
    limit?: number;
    offset?: number;
  }): Promise<Scan[]> {
    const { data } = await apiV1.get<Scan[]>('/scans/', { params });
    return data;
  },

  async get(scanId: string): Promise<Scan> {
    const { data } = await apiV1.get<Scan>(`/scans/${scanId}`);
    return data;
  },

  async create(body: ScanCreate): Promise<Scan> {
    const { data } = await apiV1.post<Scan>('/scans/', body);
    return data;
  },

  async update(scanId: string, body: ScanUpdate): Promise<Scan> {
    const { data } = await apiV1.put<Scan>(`/scans/${scanId}`, body);
    return data;
  },

  async delete(scanId: string): Promise<void> {
    await apiV1.delete(`/scans/${scanId}`);
  },

  async getMetadata(scanId: string): Promise<VolumeMetadata> {
    const { data } = await apiV1.get<VolumeMetadata>(`/viewer/scans/${scanId}/metadata`);
    return data;
  },

  async getPointClouds(scanId: string): Promise<PointCloud[]> {
    const { data } = await apiV1.get<PointCloud[]>(`/scans/${scanId}/point-clouds`);
    return data;
  },

  async importSkullBreak(body: SkullBreakImportRequest): Promise<ImportResult> {
    const { data } = await apiV1.post<ImportResult>('/scans/import-skullbreak', body);
    return data;
  },

  /** URL for vtk.js NrrdReader to fetch the raw NRRD */
  nrrdUrl(scanId: string): string {
    return viewerUrl(`/scans/${scanId}/nrrd`);
  },

  /** URL for parsed volume data (raw uint8 binary + metadata in headers) */
  volumeDataUrl(scanId: string): string {
    return viewerUrl(`/scans/${scanId}/volume-data`);
  },

  /** Fetch parsed volume data and metadata from backend.
   *  When alignTo is provided, the backend resamples this volume to match
   *  the reference scan's grid so both occupy the same world-space box. */
  async loadVolumeData(scanId: string, alignTo?: string): Promise<{
    data: Uint8Array;
    metadata: {
      dims: number[];
      spacing: number[];
      origin: number[];
      dtype: string;
      scalar_range: [number, number];
    };
  }> {
    const params = alignTo ? `?align_to=${encodeURIComponent(alignTo)}` : '';
    const response = await fetch(viewerUrl(`/scans/${scanId}/volume-data${params}`));
    if (!response.ok) throw new Error(`Failed to load volume: ${response.statusText}`);

    const metaHeader = response.headers.get('X-Volume-Metadata');
    if (!metaHeader) throw new Error('Missing volume metadata header');
    const metadata = JSON.parse(metaHeader);

    const buffer = await response.arrayBuffer();
    return { data: new Uint8Array(buffer), metadata };
  },
};
