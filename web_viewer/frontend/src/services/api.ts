import axios from 'axios';
import type { InferenceResult, FilesResponse, ConversionStatus, ApiStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

export const apiService = {
  // Health check
  async getStatus(): Promise<ApiStatus> {
    const response = await api.get<ApiStatus>('/status');
    return response.data;
  },

  // Get all results
  async getResults(): Promise<InferenceResult[]> {
    const response = await api.get<InferenceResult[]>('/results');
    return response.data;
  },

  // Get single result
  async getResult(resultId: string): Promise<InferenceResult> {
    const response = await api.get<InferenceResult>(`/results/${resultId}`);
    return response.data;
  },

  // Get files for a result
  async getResultFiles(resultId: string): Promise<FilesResponse> {
    const response = await api.get<FilesResponse>(`/results/${resultId}/files`);
    return response.data;
  },

  // Trigger conversion
  async convertResult(resultId: string, force = false, exportStl = true): Promise<any> {
    const response = await api.post(`/convert/${resultId}`, {
      force,
      export_stl: exportStl,
    });
    return response.data;
  },

  // Batch convert
  async convertBatch(force = false, exportStl = true): Promise<any> {
    const response = await api.post('/convert/batch', {
      force,
      export_stl: exportStl,
    });
    return response.data;
  },

  // Get conversion status
  async getConversionStatus(resultId: string): Promise<ConversionStatus> {
    const response = await api.get<ConversionStatus>(`/conversion-status/${resultId}`);
    return response.data;
  },

  // Get file URL
  getFileUrl(resultId: string, filename: string): string {
    return `${API_BASE_URL}/files/${resultId}/${filename}`;
  },
};

export default apiService;

