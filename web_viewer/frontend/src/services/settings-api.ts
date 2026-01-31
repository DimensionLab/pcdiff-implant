/**
 * API client for application settings.
 */
import type { AppSettings, SettingsUpdate, SystemInfo } from '../types/settings';

// Use the same base URL as the main API (goes through Vite proxy in dev)
const API_BASE = import.meta.env.VITE_API_URL || '';

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `HTTP ${response.status}`);
  }
  return response.json();
}

export const settingsApi = {
  /**
   * Get all application settings.
   */
  async getSettings(): Promise<AppSettings> {
    const response = await fetch(`${API_BASE}/api/v1/settings/`);
    return handleResponse<AppSettings>(response);
  },

  /**
   * Update application settings.
   */
  async updateSettings(updates: SettingsUpdate): Promise<AppSettings> {
    const response = await fetch(`${API_BASE}/api/v1/settings/`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updates),
    });
    return handleResponse<AppSettings>(response);
  },

  /**
   * Get system information.
   */
  async getSystemInfo(): Promise<SystemInfo> {
    const response = await fetch(`${API_BASE}/api/v1/settings/system-info`);
    return handleResponse<SystemInfo>(response);
  },
};
