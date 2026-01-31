/**
 * Type definitions for application settings.
 */

export interface AppSettings {
  inference_device: 'auto' | 'cuda' | 'mps' | 'cpu';
  default_sampling_method: 'ddim' | 'ddpm';
  default_sampling_steps: number;
  default_ensemble_count: number;
  pcdiff_model_path: string;
  voxelization_model_path: string;
}

export interface SettingsUpdate {
  inference_device?: 'auto' | 'cuda' | 'mps' | 'cpu';
  default_sampling_method?: 'ddim' | 'ddpm';
  default_sampling_steps?: number;
  default_ensemble_count?: number;
  pcdiff_model_path?: string;
  voxelization_model_path?: string;
}

export interface SystemInfo {
  cuda_available: boolean;
  mps_available: boolean;
  device_name: string | null;
  torch_version: string;
  backend_type: 'cuda' | 'cpu';
  python_version: string;
  platform: string;
}
