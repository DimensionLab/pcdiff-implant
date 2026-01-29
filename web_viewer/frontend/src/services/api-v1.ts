/**
 * Base API client for v1 endpoints.
 */
import axios from 'axios';

const API_V1_BASE = import.meta.env.VITE_API_V1_URL || '/api/v1';

export const apiV1 = axios.create({
  baseURL: API_V1_BASE,
  timeout: 60000,
});

/** Build a URL for directly serving binary data (used by vtk.js, etc.) */
export function viewerUrl(path: string): string {
  return `${API_V1_BASE}/viewer${path}`;
}
