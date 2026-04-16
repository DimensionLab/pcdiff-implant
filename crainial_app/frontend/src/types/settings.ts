/**
 * Type definitions for cran-2 application settings.
 */

export interface AppSettings {
  inference_device: 'auto' | 'cuda' | 'mps' | 'cpu';
  cran2_threshold: number;
  cloud_generation_enabled: boolean;
  runpod_endpoint_id: string;
  runpod_api_key_set: boolean; // True if API key is configured (actual key not exposed)
  aws_s3_bucket: string;
  aws_s3_region: string;
}

export interface SettingsUpdate {
  inference_device?: 'auto' | 'cuda' | 'mps' | 'cpu';
  cran2_threshold?: number;
  cloud_generation_enabled?: boolean;
  runpod_endpoint_id?: string;
  runpod_api_key?: string; // Only sent when updating, never returned
  aws_s3_bucket?: string;
  aws_s3_region?: string;
}

export interface SystemInfo {
  cuda_available: boolean;
  mps_available: boolean;
  device_name: string | null;
  torch_version: string;
  backend_type: 'cuda' | 'mps' | 'cpu';
  python_version: string;
  platform: string;
  cloud_configured: boolean;
  runpod_endpoint_id: string | null;
}
